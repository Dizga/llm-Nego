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
from peft import LoraConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import subprocess
import gc
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

torch.set_default_device('cuda')


class HfAgent:
    """
    HfAgent is an agent that utilizes HuggingFace models for causal language modeling.
    It supports training using Proximal Policy Optimization (PPO) and saving/loading models.
    """

    def __init__(
        self,
        name: str = "your_friendly_llm",
        device: str = "cuda",
        bits_and_bytes_args = None,
        lora_args = None,
        pretrained_args = None,
        

        
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

        # Initialize tokenizer
        self.tokenizer_config = # TODO
        self.training_args = TrainingArguments(**model_training_args)
        self.bits_and_bytes_configs = # TODO 
        self.lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0, bias="none") # TODO
        self.ppo_training_config = # TODO
        

    def init_ppo_trainer(self, out_directory: str, ppo_training_args: dict) -> None:
        """
        Initializes the PPO (Proximal Policy Optimization) trainer.

        Args:
            out_directory (str): The directory where training logs and checkpoints will be saved.
            ppo_training_args (dict): Arguments specific to PPO training.
        """
        os.makedirs(out_directory, exist_ok=True)
        ppo_training_args['project_kwargs'] = {'logging_dir': out_directory}
        ppo_config = PPOConfig(**ppo_training_args)

        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=ppo_config,
            tokenizer=self.tokenizer,
        )

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


    def train_ppo_json(
        self, queries: List, responses: List, scores: List[float]
    ) -> dict:
        """
        Trains the agent using PPO on a batch of queries, responses, and corresponding rewards.

        Args:
            queries (List): A list of query JSON objects.
            responses (List): A list of response JSON objects.
            scores (List): A list of rewards (scores) associated with each query-response pair.

        Returns:
            dict: A dictionary containing training statistics.
        """

        # Encode the queries into input_ids and convert the batch tensor into a list of 1D tensors
        queries_ids_tensor = self.encode_jsons(queries)['input_ids']
        queries_ids_tensor_list = [queries_ids_tensor[i] for i in range(queries_ids_tensor.size(0))]

        # Encode the responses into input_ids and convert the batch tensor into a list of 1D tensors
        responses_ids_tensor = self.encode_jsons(responses)['input_ids']
        responses_ids_tensor_list = [responses_ids_tensor[i] for i in range(responses_ids_tensor.size(0))]

        scores = [torch.tensor(s, dtype=torch.float).to(self.device) for s in scores]

        log_gpu_usage()

        # Step through PPO training 
        stats = self.ppo_trainer.step(
            queries=queries_ids_tensor_list,
            responses=responses_ids_tensor_list,
            scores=scores,
        )

        # Log stats 
        self.ppo_trainer.log_stats(
            stats=stats,
            batch={"query": queries_ids_tensor_list, "response": responses_ids_tensor_list},
            rewards=scores,
        )

        # Memory management
        del queries_ids_tensor_list, responses_ids_tensor_list, scores
        torch.cuda.empty_cache()

        return stats

    def checkpoint_model(self, model_checkp_dir: str) -> None:
        """
        Saves the model and tokenizer to a specified directory.

        Args:
            model_checkp_dir (str): The directory where the model checkpoint will be saved.
        """
        os.makedirs(model_checkp_dir, exist_ok=True)
        self.model.save_pretrained(model_checkp_dir)
        self.tokenizer.save_pretrained(model_checkp_dir)

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
         # https://huggingface.co/docs/transformers/main/en/chat_templating
        
        if self.inference_library == "hf":
            model_inputs = self.tokenizer(texts, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, do_sample=True) 
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif self.inference_library == "vllm":
            self.model.generate(texts, 
                                sampling_params=self.sampling_params, 
                                lora_request=LoRARequest("dummy_lora", 1, self.lora_weights_path)
                                )

        return responses
    

    def switch_to_hf(self, lora_weights_path=None):

        if self.inference_library == "vllm":
            # TODO
            pass
            # Empty vllm memory used

        self.inference_library = "hf"        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name
                                                          **self.pretrained_args,
                                                            quantization_config=self.quantization_conf,
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # add LoRA
        if lora_weights_path:
            self.load_lora_weights = lora_weights_path
        self.model.add_adapter(self.lora_config, adapter_name="adapter_1")
    
    def switch_to_vllm(self, lora_weights_path=None):

        if self.inference_library == "hf":
            log_gpu_usage()
            move_model_to_cpu(model)
            del model
            del ppo_trainer
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_usage()


        # Get VLLM model
        self.inference_library = "vllm"
        self.model = LLM(self.model_name, enable_lora=True)
        self.sampling_params = SamplingParams(temperature=0.7)
        if lora_weights_path:
            self.load_lora_weights = lora_weights_path
                    
    
    def save_lora_weights(self, lora_weights_path: str) -> None:
        """
        Saves only the LoRA weights to a specified directory.

        Args:
            save_directory (str): The directory where the LoRA weights will be saved.
        """
        if not isinstance(self.model, LoraModel):
            raise ValueError("The model is not a LoRA model, cannot save LoRA weights.")

        os.makedirs(lora_weights_path, exist_ok=True)
        # Save the LoRA weights only
        self.model.save_pretrained(lora_weights_path)
        self.lora_weights_path = lora_weights_path
        logging.info(f"LoRA weights saved to {lora_weights_path}")

  

