from typing import Any, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
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
    PPOTrainer
)
from peft import PeftModel, PeftConfig
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
        pretrained_args=None,
        bits_and_bytes_args=None,
        lora_args=None,
        adapter_names=None,
        generation_args=None,
        include_value_head=True,
        keep_vllm_during_training=False,
        keep_hf_during_training=True,
        keep_hf_during_eval=False,
        keep_vllm_during_eval=True,
        eval_with="vllm",
        train_with="hf",
        output_directory=None,
    ) -> None:
        """
        Initializes the HfAgent.
        """
        super().__init__()
        self.name = name
        self.device = torch.device(device) if device else torch.device("cuda")
        self.model_name = model_name
        self.include_value_head = include_value_head
        self.pretrained_args = pretrained_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_args["pretrained_model_name_or_path"]
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args) if bits_and_bytes_args else None
        self.lora_config = LoraConfig(**lora_args)
        self.hf_sampling_params = generation_args
        self.vllm_sampling_params = SamplingParams(
            temperature=generation_args["temperature"],
            top_k=-1 if generation_args["top_k"] == 0.0 else generation_args["top_k"],
            top_p=generation_args["top_p"],
            max_tokens=generation_args["max_new_tokens"],
        )
        self.adapters = {adapter_name: None for adapter_name in adapter_names}
        self.adapters['base'] = None
        self.current_adapter_name = 'base'
        self.adapter_steps = {adapter_name: 0 for adapter_name in adapter_names}
        self.hf_model = None
        self.vllm_model = None
        self.keep_vllm_during_training = keep_vllm_during_training
        self.keep_hf_during_training = keep_hf_during_training
        self.keep_hf_during_eval = keep_hf_during_eval
        self.keep_vllm_during_eval = keep_vllm_during_eval
        self.train_with = train_with
        self.eval_with = eval_with
        self.output_directory = output_directory
        
    def train(self):
        """
        Prepares the agent for training.
        """
        if not self.keep_vllm_during_training:
            self.destroy_vllm()
        if not self.keep_hf_during_training:
            self.destroy_hf()
        if self.train_with == "vllm":
            self.use_vllm_model()
        if self.train_with == "hf":
            self.use_hf_model()

    def eval(self):
        """
        Prepares the agent for evaluation.
        """
        if not self.keep_hf_during_eval:
            self.destroy_hf()
        if not self.keep_vllm_during_eval:
            self.destroy_vllm()
        if self.eval_with == "vllm":
            self.use_vllm_model()
        if self.eval_with == "hf":
            self.use_hf_model()
        

    def use_hf_model(self):
        """
        Initializes the Hugging Face model if it is not already initialized.
        """
        adapter_path = self.adapters[self.current_adapter_name]
        pretrained_args = self.pretrained_args

        if adapter_path:
            # there is already an external LoRA adapter
            if self.include_value_head:
                pretrained_args = self.pretrained_args | {'pretrained_model_name_or_path': adapter_path}
                self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    **pretrained_args, 
                    is_trainable=True, 
                    quantization_config=self.bits_and_bytes_configs
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    **self.pretrained_args, 
                    quantization_config=self.bits_and_bytes_configs
                )
                self.hf_model = PeftModel.from_pretrained(self.hf_model, adapter_path, is_trainable=True)

        else:
            # no external LoRA adapter
            if self.include_value_head:
                self.hf_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    **pretrained_args, 
                    is_trainable=True, 
                    quantization_config=self.bits_and_bytes_configs
                )
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    **self.pretrained_args, 
                    quantization_config=self.bits_and_bytes_configs
                )
                self.hf_model = get_peft_model(self.hf_model, self.lora_config)
                self.hf_model.train()

    def destroy_hf(self):
        """
        Destroys the Hugging Face model to free up memory.
        """
        if self.hf_model is not None:
            self.export_current_adapter()
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

    def log_gpu_usage(self, message: str) -> None:
        """
        Logs the GPU memory usage.

        Args:
            message (str): A message to include in the log.
        """
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        logging.info(f"{message}: GPU memory allocated: {gpu_memory:.2f} GB")



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

        if self.eval_with == "vllm":


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
    
    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the current adapter for the model.

        Args:
            adapter_name (str): The name of the adapter to set.
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found in available adapters.")
        
        self.current_adapter_name = adapter_name
        logging.info(f"Adapter set to '{adapter_name}'.")

        # Load the adapter if necessary
        adapter_path = self.adapters[adapter_name]
        if adapter_path and self.hf_model is not None:
            self.use_hf_model()  # Ensure the model is using the correct adapter

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

        # Save only the LoRA weights
        if isinstance(self.hf_model, PeftModel):
            self.hf_model.save_pretrained(adapter_path)
        else:
            logging.warning("Model is not a LoraModel, skipping LoRA weights saving.")

        # For vllm
        with open(os.path.join(adapter_path, "config.json"), "w") as f:
            json.dump({"model_type": "gpt2"}, f)

        # Update the adapter path after export
        self.adapters[self.current_adapter_name] = adapter_path
        logging.info(f"LoRA weights saved to {adapter_path}")
