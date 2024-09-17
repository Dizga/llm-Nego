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
from trl import (
    SFTTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from peft import LoraConfig, LoraModel
import os

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
        generation_args=None
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
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args)
        self.lora_config = LoraConfig(**lora_args) 
        self.ppo_training_args = ppo_trainer_args


        self.vllm_sampling_params = SamplingParams(
            **generation_args         
        )

        self.hf_sampling_params = copy.deepcopy(generation_args)
        self.hf_sampling_params['max_new_tokens'] = self.hf_sampling_params.pop('max_tokens')


        self.inference_library = None
        self.ppo_trainer = None
        self.lora_pretrained_path = lora_pretrained_path


        
        

    def encode_jsons(self, data: List[dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encodes JSON conversation data into tensors, ensuring proper padding and attention masks.

        Args:
            data (List[Dict]): A list of JSON objects representing conversations.

        Returns:
        """

        formatted = self.tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=False)
        
        # Ensure a pad_token is set for the tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize with padding and truncation
        tokenized = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)


        return tokenized
    

    def train_ppo(
            self, path, queries: List, responses: List, scores: List[float]
        ) -> dict:
        """
        Trains the agent using PPO on a batch of queries, responses, and corresponding rewards.

        Returns:
            dict: A dictionary containing training statistics.
        """

        self.switch_to_hf()

        ds = len(queries)  # get datasize
        if ds == 0:
            logging.warning("train_ppo received empty dataset")
            self.ppo_trainer = None
            return

        # adjust sizes with respect to parameters and ds
        self.ppo_training_args['batch_size'] = min(self.ppo_training_args['batch_size'], ds)
        self.ppo_training_args['gradient_accumulation_steps'] = self.ppo_training_args['batch_size']

        self.ppo_training_args['project_kwargs'] = {'logging_dir': os.path.join(path, self.name + '_ppo_tensorboard')}


        # TODO
        self.ppo_trainer = ReinforceTrainer(
            model=self.model,
            ref_model=self.model,
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

            # Log the size of the current batch
            logging.info(f"Batch size: {len(batch_queries)} queries, {len(batch_responses)} responses.")

            # Encode the queries into input_ids and convert the batch tensor into a list of 1D tensors
            queries_ids_tensor = self.encode_jsons(batch_queries)['input_ids']
            queries_ids_tensor_list = [queries_ids_tensor[i] for i in range(queries_ids_tensor.size(0))]

            # Encode the responses into input_ids and attention masks
            tokenized_responses = self.encode_jsons(batch_responses)
            responses_ids_tensor = tokenized_responses['input_ids']
            responses_ids_tensor_list = [responses_ids_tensor[i] for i in range(responses_ids_tensor.size(0))]

            # Use the attention masks for the responses
            response_masks = tokenized_responses['attention_mask']
            response_masks_list = [response_masks[i] for i in range(response_masks.size(0))]

            # Convert batch scores to tensors
            batch_tensor_scores = [torch.tensor(s, dtype=torch.float).to(self.device) for s in batch_scores]

            # Log the start of PPO trainer step
            logging.info(f"Starting PPO step for batch {b+1}/{nb_batches}...")

            # Step through PPO training 
            stats = self.ppo_trainer.step(
                queries=queries_ids_tensor_list,
                responses=responses_ids_tensor_list,
                scores=batch_tensor_scores,
                response_masks=response_masks_list  # Fill in the TODO here
            )

            # Log statistics and rewards
            logging.info(f"PPO step for batch {b+1}/{nb_batches} completed. Logging stats...")

            # Log the training stats for the current batch
            self.ppo_trainer.log_stats(
                stats=stats,
                batch={"query": queries_ids_tensor_list, "response": responses_ids_tensor_list},
                rewards=batch_tensor_scores,
            )

            del queries_ids_tensor_list, responses_ids_tensor_list, response_masks_list, batch_tensor_scores
            torch.cuda.empty_cache()

            # Calculate and log the time taken for this batch
            batch_duration = time.time() - start_time
            logging.info(f"Batch {b+1}/{nb_batches} training completed in {batch_duration:.2f} seconds.")

        logging.info("PPO training completed for all batches.")

        # Ensure garbage collection is performed
        self.delete_tensor_list(queries)
        self.delete_tensor_list(responses)
        self.delete_tensor_list(scores)
        torch.cuda.empty_cache()

        self.save_lora_weights(os.path.join(path, self.name + '_lora_weights'))


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

        if len(contexts) == 0:
            return []
        
        # Generate with Hugging Face
        if self.inference_library == "hf":
            model_inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, 
                                                    **self.hf_sampling_params) 
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Generate with VLLM
        elif self.inference_library == "vllm":
            if self.lora_pretrained_path:
                responses = self.model.generate(texts, 
                                    sampling_params=self.vllm_sampling_params, 
                                    lora_request=LoRARequest("dond_lora", 1, self.lora_pretrained_path)
                                    )
            else:
                responses = self.model.generate(texts, 
                    sampling_params=self.vllm_sampling_params
                    )
            responses = [response.outputs[0].text for response in responses]


        return responses
    

    def switch_to_hf(self):

        # Free GPU memory taken by VLLM
        if self.inference_library == "vllm":
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        elif self.inference_library == "hf":
            return 

        self.inference_library = "hf"        

        if self.lora_pretrained_path:
            self.pretrained_args['pretrained_model_name_or_path'] = self.lora_pretrained_path

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                                                          **self.pretrained_args,
                                                            quantization_config=self.bits_and_bytes_configs,
                                                            peft_config=self.lora_config
                                                          )


    def switch_to_vllm(self):

        # Free GPU memory taken by Hugging Face
        if self.inference_library == "hf":
            move_model_to_cpu(self.model)
            del self.model
            del self.ppo_trainer
            gc.collect()
            torch.cuda.empty_cache()

        elif self.inference_library == "vllm":
            return 
        

        # Get VLLM model
        self.inference_library = "vllm"
        if self.lora_pretrained_path:
            self.model = LLM(self.model_name, enable_lora=True)
        else:
            self.model = LLM(self.model_name, enable_lora=False)
            
        
    def save_lora_weights(self, lora_weights_path: str) -> None:
        """
        Saves only the LoRA weights to a specified directory.

        Args:
            save_directory (str): The directory where the LoRA weights will be saved.
        """

        os.makedirs(lora_weights_path, exist_ok=True)
        self.ppo_trainer.save_pretrained(lora_weights_path)
        self.lora_pretrained_path = lora_weights_path
        logging.info(f"LoRA weights saved to {lora_weights_path}")

    

