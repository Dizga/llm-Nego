from typing import Any, Tuple, List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
import shutil
from trl import (
    SFTTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from peft import LoraConfig, LoraModel
import os
from functools import partial

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from utils.log_gpu_usage import log_gpu_usage
from utils.model_to_cpu import move_model_to_cpu
import logging
import time
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import subprocess
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from omegaconf import OmegaConf

from training.custom_ppo_trainer import CustomPPOTrainer
from training.reinforce_trainer import ReinforceTrainer
torch.set_default_device('cuda')
import copy
import numpy as np
import hydra

class HfAgent:
    """
    HfAgent is an agent that utilizes HuggingFace models for causal language modeling.
    It supports training using Proximal Policy Optimization (PPO) and saving/loading models.
    """

    def __init__(
        self,
        name: str = "your_friendly_llm",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",
        pretrained_args = None,
        bits_and_bytes_args = None,
        lora_args = None,
        ppo_trainer_args = None,
        save_lora_weights= None,
        lora_pretrained_path=None,
        generation_args=None,
        default_training_mode='ppo',
        keep_vllm_during_training=False,
        keep_hf_during_generation=True,
        generate_with="vllm",
        ppo_trainer_class=None
    ) -> None:
        """
        Initializes the HfAgent.

        Args:
            name (str): The name of the agent.
            device (str): The device to run the model on, either 'cuda' or 'cpu'.
            tokenizer_name (str): The tokenizer to be used, specified by the tokenizer name or path.
            inherit_model (bool): If True, load a pre-existing model; otherwise, initialize a new model.
            model_args (dict): Arguments for loading the model.
            bits_and_bytes_args (dict): Configuration for quantization (optional).
            lora_args (dict): LoRA (Low-Rank Adaptation) configuration for model fine-tuning.
            model_training_args (dict): Training arguments for fine-tuning the model.
            out_folder (str): The output folder for saving models and logs.
        """
        super().__init__()
        self.name = name
        self.device = torch.device(device)

        self.model_name = model_name
        self.pretrained_args = pretrained_args
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_args['pretrained_model_name_or_path'])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args)
        self.lora_config = LoraConfig(**lora_args) 
        self.ppo_training_args = ppo_trainer_args
        self.ppo_trainer_class = ppo_trainer_class

        #self.sft_config = SFTConfig(output_dir=os.path.join(out_dir, 'sft_lora_model'), packing=True)

        self.hf_sampling_params = generation_args

        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args['temperature'],
            top_k=generation_args['top_k'],
            top_p=generation_args['top_p'],
            max_tokens=generation_args['max_new_tokens']
        )

        self.default_training_mode = default_training_mode
        self.inference_library = None
        self.ppo_trainer = None
        self.lora_pretrained_path = lora_pretrained_path

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_directory = hydra_cfg['runtime']['output_dir']
        self.adapter_id = 1
        self.hf_model = None
        self.vllm_model = None

        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_hf_during_generation = keep_hf_during_generation
        self.generate_with = generate_with
        


    def batch_encode(self, data: List[List[dict]], pad=False, is_response=False) -> ...:

        if is_response:
            formatted = [d[0]['content'] + self.tokenizer.eos_token for d in data]
            self.tokenizer.padding_side = "right"
        else:
            formatted = self.tokenizer.apply_chat_template(data, 
                                                           tokenize=False, 
                                                           add_generation_prompt=True)
            self.tokenizer.padding_side = "left"

        # Tokenize with padding and truncation
        if pad:
            tokenized = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            self.tokenizer.padding_side = "left"

            return list(tokenized['input_ids'])
    
        else:
            return [self.tokenizer(d, return_tensors="pt").to(self.device)['input_ids'].squeeze() for d in formatted]
    

    def train_ppo(
            self, queries: List[List[dict]], responses: List[List[dict]], scores: List[float]
        ) -> None:
        """
        Trains the agent using PPO on a batch of queries, responses, and corresponding rewards.

        Returns:
            dict: A dictionary containing training statistics.
        """

        if not self.keep_vllm_during_training: 
            if self.vllm_model != None:
                del self.vllm_model
                gc.collect()
                torch.cuda.empty_cache()
                self.vllm_model = None

        self.use_hf_model()
        

        ds = len(queries)  # get datasize
        if ds == 0:
            logging.warning("train_ppo received empty dataset")
            self.ppo_trainer = None
            return

        # adjust sizes with respect to parameters and ds
        self.ppo_training_args['batch_size'] = min(self.ppo_training_args['batch_size'], ds)
        self.ppo_training_args['gradient_accumulation_steps'] = self.ppo_training_args['batch_size']

        self.ppo_training_args['project_kwargs'] = {'logging_dir': os.path.join(self.output_directory, self.name + '_ppo_tensorboard')}


        self.ppo_trainer = globals()[self.ppo_trainer_class](
            model=self.hf_model,
            ref_model=None,
            config=PPOConfig(**self.ppo_training_args),
            tokenizer=self.tokenizer,
        )

        bs = self.ppo_training_args['batch_size']
        nb_batches = ds // bs
        logging.info(f"Starting PPO training with {ds} samples, batch size {bs}, total batches {nb_batches}")

        # Start training process
        for b in range(nb_batches):
            start_time = time.time()

            logging.info(f"Training on batch {b+1}/{nb_batches} started.")

            beg, end = (b * bs, (b + 1) * bs)
            batch_queries = queries[beg:end]
            batch_responses = responses[beg:end]
            batch_scores = scores[beg:end]

            logging.info(f"Batch size: {len(batch_queries)} queries, {len(batch_responses)} responses.")

            encoded_batch_queries = self.batch_encode(batch_queries)
            encoded_batch_responses = self.batch_encode(batch_responses, is_response=True)
            encoded_batch_scores = [torch.tensor(s, dtype=torch.float).to(self.device) for s in batch_scores]
            assert len(encoded_batch_queries) == len(encoded_batch_responses)

            decoded_queries = [self.tokenizer.decode(q, skip_special_tokens=False) for q in encoded_batch_queries]
            logging.debug(f"Decoded Queries: {decoded_queries}")
            decoded_responses = [self.tokenizer.decode(r, skip_special_tokens=False) for r in encoded_batch_responses]
            logging.info(f"Decoded Responses: {decoded_responses}")
            

            logging.info(f"Starting PPO step for batch {b+1}/{nb_batches}...")
            stats = self.ppo_trainer.step(
                queries=encoded_batch_queries,
                responses=encoded_batch_responses,
                scores=encoded_batch_scores
            )

            logging.info(f"PPO step for batch {b+1}/{nb_batches} completed. Logging stats...")

            self.ppo_trainer.log_stats(
                stats=stats,
                batch={"query": encoded_batch_queries, 
                    "response": encoded_batch_responses,
                    "ref_rewards": encoded_batch_scores
                    },
                columns_to_log=["query", "response", "ref_rewards"],
                rewards=encoded_batch_scores,
            )

            self.delete_tensor_list(encoded_batch_queries)
            self.delete_tensor_list(encoded_batch_responses)
            self.delete_tensor_list(encoded_batch_scores)

            batch_duration = time.time() - start_time
            logging.info(f"Batch {b+1}/{nb_batches} training completed in {batch_duration:.2f} seconds.")

        logging.info("PPO training completed for all batches.")

        # Ensure garbage collection is performed
        self.delete_tensor_list(queries)
        self.delete_tensor_list(responses)
        self.delete_tensor_list(scores)
        del self.ppo_trainer
        del stats
        gc.collect()
        torch.cuda.empty_cache()

        self.save_lora_weights()

    def train_sft(self, dataset_path):
        """
        Dataset should have "conversational" form (see # https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support)
        """
        self.use_hf_model()
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        sft_trainer = SFTTrainer(
            self.hf_model,
            args=self.sft_config,
            train_dataset=dataset,
        )
        sft_trainer.train()

    def delete_tensor_list(self, tensor_list: List[Any]) -> None:
        """
        Deletes tensors from a list to free up memory.

        Args:
            tensor_list (List[Any]): List of tensors to delete.
        """
        for tensor in tensor_list:
            del tensor

    def prompt(self, contexts) -> str:
        """

        Returns:
            str: The generated response from the model.
        """

        texts = self.tokenizer.apply_chat_template(contexts, tokenize=False, add_generation_prompt=True)

        if len(contexts) == 0: return []
        
        # Generate with VLLM
        if self.generate_with == "vllm":

            if not self.keep_hf_during_generation:
                del self.hf_model
                gc.collect()
                torch.cuda.empty_cache()

            self.use_vllm_model()
            
            with torch.no_grad():
                if self.lora_pretrained_path:
                    logging.info('Generating using VLLM (with LoRA)')
                    decoded = self.vllm_model.generate(texts, 
                                        sampling_params=self.vllm_sampling_params, 
                                        lora_request=LoRARequest("dond_lora", 
                                                                1, 
                                                                self.lora_pretrained_path)
                                        )
                else:
                    logging.info('Generating using VLLM (without LoRA)')
                    decoded = self.vllm_model.generate(texts, 
                        sampling_params=self.vllm_sampling_params
                        )
            responses = [d.outputs[0].text for d in decoded]
            del decoded # TODO : verify this does not break everything
        
        # Generate with Hugging Face
        elif self.generate_with == "hf":
            logging.info('Generating using Hugging Face')
            model_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.hf_model.generate(**model_inputs, 
                                                    **self.hf_sampling_params) 
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        else:
            logging.warning('No model in hf agent!')


        return responses
    

    def use_hf_model(self):

        if self.hf_model == None and self.lora_pretrained_path:
            self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                                                                self.lora_pretrained_path,
                                                                is_trainable=True
                                                                )
        elif self.hf_model == None:

            if self.default_training_mode == 'ppo':
                self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                                                                **self.pretrained_args,
                                                                    quantization_config=self.bits_and_bytes_configs,
                                                                    peft_config=self.lora_config
                                                                )
            elif self.default_training_mode == 'sft':
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                                                    **self.pretrained_args,
                                                        quantization_config=self.bits_and_bytes_configs,
                                                        peft_config=self.lora_config
                                                    )


    def use_vllm_model(self):

        if self.lora_pretrained_path and self.vllm_model == None:
            del self.vllm_model
            gc.collect()
            torch.cuda.empty_cache()
            self.vllm_model = LLM(self.model_name, enable_lora=True, max_lora_rank=128)

        elif self.vllm_model == None:
            gc.collect()
            torch.cuda.empty_cache()
            self.vllm_model = LLM(self.model_name, enable_lora=False, max_lora_rank=128)

    def save_lora_weights(self) -> None:
        """
        Saves only the LoRA weights to a specified directory. If the directory
        already exists, it deletes the existing directory before saving.

        Args:
            save_directory (str): The directory where the LoRA weights will be saved.
        """

        # Construct the path for LoRA weights
        lora_weights_path = os.path.join(self.output_directory, self.name + '_lora_weights')
        
        # If the folder exists, delete it
        if os.path.exists(lora_weights_path):
            shutil.rmtree(lora_weights_path)
            logging.info(f"Existing directory '{lora_weights_path}' deleted.")

        os.makedirs(lora_weights_path, exist_ok=True)
        logging.info(f"Directory '{lora_weights_path}' created.")

        # Save the LoRA weights
        self.hf_model.save_pretrained(lora_weights_path)
        
        # Update the path attribute
        self.lora_pretrained_path = lora_weights_path
        logging.info(f"LoRA weights saved to {lora_weights_path}")


    

