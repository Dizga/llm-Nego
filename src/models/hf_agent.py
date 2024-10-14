from typing import Any, Tuple, List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import os
os.environ["WANDB_DISABLED"] = "True"
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
from vllm.distributed.parallel_state import destroy_model_parallel
from omegaconf import OmegaConf
import copy
import numpy as np
import hydra
from transformers import Trainer
import json
from training.custom_ppo_trainers import NoValuePPOTrainer

class HfAgent:
    """
    HfAgent is an agent that utilizes HuggingFace models for causal language modeling.
    It supports training using Proximal Policy Optimization (PPO) and saving/loading models.
    """

    def __init__(
        self,
        name: str = "your_friendly_llm",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",  # Ensure default is set to "cuda"
        pretrained_args=None,
        bits_and_bytes_args=None,
        lora_args=None,
        ppo_trainer_args=None,
        lora_pretrained_path=None,
        adapter_names=None,
        generation_args=None,
        default_training_mode="ppo",
        keep_vllm_during_training=False,
        keep_hf_during_generation=True,
        destroy_ppo_trainer_after_training=True,
        generate_with="vllm",
        ppo_trainer_class=None,
        sft_args=None,  # Add this parameter
    ) -> None:
        """
        Initializes the HfAgent.

        Args:
            name (str): The name of the agent.
            model_name (str): The name of the model to be used.
            device (str): The device to run the model on, either 'cuda' or 'cpu'.
            pretrained_args (dict): Arguments for loading the pretrained model.
            bits_and_bytes_args (dict): Configuration for quantization (optional).
            lora_args (dict): LoRA (Low-Rank Adaptation) configuration for model fine-tuning.
            ppo_trainer_args (dict): Training arguments for PPO.
            export_current_adapter (str): Path to save LoRA weights.
            lora_pretrained_path (str): Path to pretrained LoRA weights.
            generation_args (dict): Arguments for text generation.
            default_training_mode (str): Default training mode, either 'ppo' or 'sft'.
            keep_vllm_during_training (bool): Whether to keep VLLM during training.
            keep_hf_during_generation (bool): Whether to keep HF during generation.
            generate_with (str): Library to use for generation, either 'vllm' or 'hf'.
            ppo_trainer_class (str): Class name for PPO trainer.
            sft_config (dict): Configuration for Supervised Fine-Tuning (SFT).
        """
        super().__init__()
        self.name = name
        self.device = torch.device(device) if device else torch.device("cuda")  # Ensure device is set

        self.model_name = model_name
        self.pretrained_args = pretrained_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_args["pretrained_model_name_or_path"]
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args) if bits_and_bytes_args else None
        self.lora_config = LoraConfig(**lora_args)
        self.ppo_training_args = ppo_trainer_args
        self.ppo_trainer_class = ppo_trainer_class

        self.hf_sampling_params = generation_args
        vllm_samp_args = {
            "temperature": generation_args["temperature"],
            "top_k": -1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            "top_p": generation_args["top_p"],
            "max_tokens": generation_args["max_new_tokens"],
        }

        self.vllm_sampling_params = SamplingParams(**vllm_samp_args)
        logging.info(f"VLLM sampling params: {self.vllm_sampling_params}")

        self.default_training_mode = default_training_mode
        self.inference_library = None
        self.ppo_trainer = None
        adapter_path = lora_pretrained_path

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        self.output_directory = hydra_cfg["runtime"]["output_dir"]
        self.hf_model = None
        self.vllm_model = None

        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_hf_during_generation = keep_hf_during_generation
        self.destroy_ppo_trainer_after_training = destroy_ppo_trainer_after_training
        self.generate_with = generate_with

        self.sft_config = SFTConfig(**sft_args, output_dir=self.output_directory+"/") # Initialize sft_config

        self.adapters = {adapter_name: None for adapter_name in adapter_names}
        self.current_adapter_name = 'base'



    def batch_encode(self, data: List[List[dict]], pad=False, is_response=False) -> ...:
        """
        Encodes a batch of data using the tokenizer.

        Args:
            data (List[List[dict]]): The data to encode.
            pad (bool): Whether to pad the sequences.
            is_response (bool): Whether the data is a response.

        Returns:
            List[torch.Tensor]: The encoded data.
        """
        if is_response:
            formatted = [d[0]["content"] + self.tokenizer.eos_token for d in data]
            self.tokenizer.padding_side = "right"
        else:
            formatted = self.tokenizer.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True
            )
            self.tokenizer.padding_side = "left"

        if pad:
            tokenized = self.tokenizer(
                formatted, return_tensors="pt", padding=True
            ).to(self.device)
            self.tokenizer.padding_side = "left"
            return list(tokenized["input_ids"])
        else:
            return [
                self.tokenizer(d, return_tensors="pt", add_special_tokens=False)
                .to(self.device)["input_ids"]
                .squeeze()
                for d in formatted
            ]

    def train_ppo(
        self, queries: List[List[dict]], responses: List[List[dict]], scores: List[float]
    ) -> None:
        """
        Trains the agent using PPO on a batch of queries, 
        responses, and corresponding rewards.

        Args:
            queries (List[List[dict]]): The queries for training.
            responses (List[List[dict]]): The responses for training.
            scores (List[float]): The scores for training.
        """
        if not self.keep_vllm_during_training:
            self.destroy_vllm()

        self.use_hf_model()

        ds = len(queries)
        if ds == 0:
            logging.warning("train_ppo received empty dataset")
            self.ppo_trainer = None
            return
        

        if self.ppo_training_args["batch_size"] == -1:
            self.ppo_training_args["batch_size"] = ds

        else: 
            self.ppo_training_args["batch_size"] = min(
            self.ppo_training_args["batch_size"], ds
        )
            
        self.ppo_training_args["gradient_accumulation_steps"] = self.ppo_training_args[
            "batch_size"
        ] // self.ppo_training_args["mini_batch_size"]

        self.ppo_training_args["project_kwargs"] = {
            "logging_dir": os.path.join(
                self.output_directory, self.name + "_ppo_tensorboard"
            )
        }

        if self.ppo_trainer is None:
            self.ppo_trainer = globals()[self.ppo_trainer_class](
                model=self.hf_model,
                ref_model=self.hf_model,
                config=PPOConfig(**self.ppo_training_args),
                tokenizer=self.tokenizer,
            )
        else:
            self.ppo_trainer.model = self.hf_model
            self.ppo_trainer.ref_model = self.hf_model

        bs = self.ppo_training_args["batch_size"]
        nb_batches = ds // bs
        logging.info(
            f"Starting PPO training with {ds} samples, \
            batch size {bs}, total batches {nb_batches}"
        )

        for b in range(nb_batches):
            start_time = time.time()

            logging.info(f"Training on batch {b+1}/{nb_batches} started.")

            beg, end = (b * bs, (b + 1) * bs)
            batch_queries = queries[beg:end]
            batch_responses = responses[beg:end]
            batch_scores = scores[beg:end]

            logging.info(
                f"Batch size: {len(batch_queries)} queries, {len(batch_responses)} responses."
            )

            encoded_batch_queries = self.batch_encode(batch_queries)
            encoded_batch_responses = self.batch_encode(batch_responses, is_response=True)
            encoded_batch_scores = [
                torch.tensor(s, dtype=torch.float).to(self.device) for s in batch_scores
            ]
            assert len(encoded_batch_queries) == len(encoded_batch_responses)

            logging.info(f"Starting PPO step for batch {b+1}/{nb_batches}...")
            stats = self.ppo_trainer.step(
                queries=encoded_batch_queries,
                responses=encoded_batch_responses,
                scores=encoded_batch_scores,
            )

            logging.info(
                f"PPO step for batch {b+1}/{nb_batches} completed. Logging stats..."
            )

            self.ppo_trainer.log_stats(
                stats=stats,
                batch={
                    "query": encoded_batch_queries,
                    "response": encoded_batch_responses,
                    "ref_rewards": encoded_batch_scores,
                },
                columns_to_log=["query", "response", "ref_rewards"],
                rewards=encoded_batch_scores,
            )

            # Export batch to JSON
            self.export_batch_to_json(encoded_batch_queries, encoded_batch_responses, encoded_batch_scores, b)
            self.delete_tensor_list(encoded_batch_queries)
            self.delete_tensor_list(encoded_batch_responses)
            self.delete_tensor_list(encoded_batch_scores)
            gc.collect()
            torch.cuda.empty_cache()

            batch_duration = time.time() - start_time
            logging.info(
                f"Batch {b+1}/{nb_batches} training \
                completed in {batch_duration:.2f} seconds."
            )

        logging.info("PPO training completed for all batches.")

        self.export_current_adapter()
        self.delete_tensor_list(queries)
        self.delete_tensor_list(responses)
        self.delete_tensor_list(scores)
        if self.destroy_ppo_trainer_after_training:
            del self.ppo_trainer
        #del self.ppo_trainer.model
        #del self.ppo_trainer.ref_model
        del stats
        gc.collect()
        torch.cuda.empty_cache()
        if self.destroy_ppo_trainer_after_training:
            self.ppo_trainer = None


    def export_batch_to_json(self, queries, responses, scores, batch_number):
        """
        Exports the batch data to a JSON file.

        Args:
            queries (List[torch.Tensor]): The queries for the batch.
            responses (List[torch.Tensor]): The responses for the batch.
            scores (List[torch.tensor]): The scores for the batch.
            batch_number (int): The batch number.
        """
        export_data = []
        for q, r, s in zip(queries, responses, scores):
            decoded_query = self.tokenizer.decode(q, skip_special_tokens=False)
            decoded_response = self.tokenizer.decode(r, skip_special_tokens=False)
            export_data.append({"query": decoded_query, "response": decoded_response, "score": s.item()})

        export_dir = os.path.join(self.output_directory, "ppo_batches")
        os.makedirs(export_dir, exist_ok=True)
        export_file = os.path.join(export_dir, f"batch_{batch_number}_run_{int(time.time())}.json")

        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=4)

        logging.info(f"Batch {batch_number} exported to {export_file}")

    def train_sft(self, dataset_path):
        """
        Trains the agent using Supervised Fine-Tuning (SFT) on a dataset.

        Args:
            dataset_path (str): Path to the dataset for training.
        """
        if not self.keep_vllm_during_training:
            self.destroy_vllm()
        self.use_hf_model()

        # WARNING: THIS WILL BUG IF YOU USE TORCH.DEFAULT_DEVICE ANYWHERE ELSE IN THE CODE (not
        # your fault, just hf)

        
        # Create Dataset from dictionary
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        sft_trainer = SFTTrainer(
            model=self.hf_model,
            args=self.sft_config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        logging.info(f"Model device: {next(sft_trainer.model.parameters()).device}")
        logging.info(f"Trainer device: {sft_trainer.args.device}")

        sft_trainer.train()

        # Save LoRA weights after training
        self.export_current_adapter()

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
        Generates a response from the model based on the provided contexts.

        Args:
            contexts (List[dict]): The contexts for generation.

        Returns:
            str: The generated response from the model.
        """

        adapter_path = self.adapters[self.current_adapter_name]

        if len(contexts) == 0: return []

        texts = self.tokenizer.apply_chat_template(
            contexts, tokenize=False, add_generation_prompt=True
        )

        if self.generate_with == "vllm":
            if not self.keep_hf_during_generation:
                self.destroy_hf()

            self.use_vllm_model()

            with torch.no_grad():
                if adapter_path is not None:
                    logging.info(f"Generating using VLLM (with LoRA at {adapter_path})")
                    request = LoRARequest("dond_lora", 1, adapter_path)
                    decoded = self.vllm_model.generate(
                        texts,
                        sampling_params=self.vllm_sampling_params,
                        lora_request=request,
                    )
                    del request
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    logging.info("Generating using VLLM (without LoRA)")
                    decoded = self.vllm_model.generate(
                        texts, sampling_params=self.vllm_sampling_params
                    )

            responses = [d.outputs[0].text for d in decoded]

            del decoded
            gc.collect()
            torch.cuda.empty_cache()

        elif self.generate_with == "hf":
            self.use_hf_model()

            logging.info("Generating using Hugging Face")
            model_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                generated_ids = self.hf_model.generate(
                    **model_inputs, **self.hf_sampling_params
                )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(
                    model_inputs["input_ids"], generated_ids
                )
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        else:
            logging.warning("No model in hf agent!")

        return responses
    
    def set_adapter(self, name: str) -> None:
        """
        Set the current LoRA adapter to the specified name.

        Args:
            name (str): The name of the adapter to switch to.
        """
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not found in {self.name}")

        self.current_adapter_name = name
        adapter_path = self.adapters[name]

        if adapter_path:
            logging.info(f"Switched to adapter: {name} at {adapter_path}")
        else:
            logging.info(f"Switched to adapter: {name} Loading base model.")

    def use_hf_model(self):
        """
        Initializes the Hugging Face model if it is not already initialized.
        """
        # TODO: fix not changing adapter if not none
        adapter_path = self.adapters[self.current_adapter_name]

        if adapter_path:

            pretrained_args = self.pretrained_args | {'pretrained_model_name_or_path': adapter_path}

            if self.default_training_mode == "ppo":
                logging.info(f"Loading LoRA weights for PPO from {adapter_path}")
                self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    **pretrained_args, 
                    is_trainable=True, 
                    quantization_config=self.bits_and_bytes_configs
                )

            elif self.default_training_mode == "sft":
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    **self.pretrained_args,
                    quantization_config=self.bits_and_bytes_configs,
                    device_map="auto"
                )
                self.hf_model = PeftModel.from_pretrained(
                    self.hf_model, adapter_path, is_trainable=True
                )
                self.hf_model.print_trainable_parameters()

        elif self.default_training_mode == "ppo":

            self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                **self.pretrained_args,
                quantization_config=self.bits_and_bytes_configs,
                peft_config=self.lora_config,
            )

        elif self.default_training_mode == "sft":

            self.hf_model = AutoModelForCausalLM.from_pretrained(
                **self.pretrained_args,
                quantization_config=self.bits_and_bytes_configs,
            )
            self.hf_model = get_peft_model(self.hf_model, self.lora_config)
            self.hf_model.print_trainable_parameters()


    def destroy_hf(self):
        """
        Destroys the Hugging Face model to free up memory.
        """
        if self.hf_model is not None:
            self.log_gpu_usage("Before destroying HF.")
            del self.hf_model
            gc.collect()
            torch.cuda.empty_cache()
            self.hf_model = None
            self.log_gpu_usage("After destroying HF.")

    def use_vllm_model(self):
        """
        Initializes the VLLM model if it is not already initialized.
        """
        adapter_path = self.adapters[self.current_adapter_name]

        # Check if the adapter path exists
        enable_lora = adapter_path is not None and os.path.exists(adapter_path)

        if self.vllm_model is None:
            gc.collect()
            torch.cuda.empty_cache()
            self.vllm_model = LLM(self.model_name, enable_lora=enable_lora, max_lora_rank=256)

    def destroy_vllm(self):
        """
        Destroys the VLLM model to free up memory.
        """
        if self.vllm_model is not None:
            self.log_gpu_usage("Before destroying VLLM")
            del self.vllm_model
            gc.collect()
            torch.cuda.empty_cache()
            self.vllm_model = None
            self.log_gpu_usage("After destroying VLLM.")

    def export_current_adapter(self) -> None:
        """
        Saves only the LoRA weights to a specified directory. If the directory
        already exists, it deletes the existing directory before saving.
        """
        adapter_path = os.path.join(self.output_directory, self.current_adapter_name)

        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
            logging.info(f"Existing directory '{adapter_path}' deleted.")

        os.makedirs(adapter_path, exist_ok=True)

        if self.ppo_trainer is not None:
            self.ppo_trainer.save_pretrained(adapter_path)
        else:
            self.hf_model.save_pretrained(adapter_path)

        # Update the adapter path after export
        self.adapters[self.current_adapter_name] = adapter_path
        logging.info(f"LoRA weights saved to {adapter_path}")

    def log_gpu_usage(self, message: str) -> None:
        """
        Logs the GPU memory usage.

        Args:
            message (str): A message to include in the log.
        """
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"{message}: GPU memory allocated: {gpu_memory:.2f} GB")