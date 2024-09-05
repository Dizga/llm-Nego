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
import logging
import time
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_default_device('cuda')


class HfAgent:
    """
    HfAgent is an agent that utilizes HuggingFace models for causal language modeling.
    It supports training using Proximal Policy Optimization (PPO) and saving/loading models.
    """

    def __init__(
        self,
        name: str = "agent",
        device: str = "cuda",  # 'cuda' or 'cpu'
        tokenizer_name: str = "microsoft/Phi-3-mini-128k-instruct",
        model_training_args: dict = None,
        out_folder: str = "checkpoints",
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Set training arguments
        if model_training_args:
            self.training_args = TrainingArguments(**model_training_args)

        self.out_folder = out_folder

        self.history = []

    def _initialize_model(self, pretrained_args: dict, bits_and_bytes_args: dict, lora_args: dict):
        """Initializes the model with LoRA and quantization configurations."""

        # Initialize the quantization configuration
        self.quantization_conf = BitsAndBytesConfig(**bits_and_bytes_args)

        # Initialize the LoRA configuration
        self.lora_config = LoraConfig(**lora_args)
        
        # Load the model with quantization
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            **pretrained_args,
            quantization_config=self.quantization_conf,
            peft_config=self.lora_config
        )

        # Compile for faster generation (https://github.com/huggingface/huggingface-llama-recipes/blob/main/torch_compile.py)
        # self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
        # self.model.generation_config.cache_implementation = "static"
        # self.model.generation_config.max_length = 128



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

    def reset_messages(self) -> None:
        """
        Resets the conversation history.
        """
        self.history = []

    def set_error_last_message(self):
        self.history[-1]["is_error"] = True

    def add_message(self, role: str, message: str, is_error: bool = False, is_new_round: bool = False) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            message (str): The message content.
            is_error (bool): Indicates if the message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.
        """

        self.history.append({
            "role": role,
            "content": message,
            "is_error": is_error,
            "is_new_round": is_new_round
        })

    def add_system_message(self, message: str) -> None:
        """
        Adds a system message to the conversation history.

        Args:
            message (str): The system message content.
        """
        self.add_message("system", message)



    def prompt(self, message: str, is_error: bool = False, is_new_round: bool = False, model_args: dict = None) -> str:
        """
        Adds a user message to the conversation history and generates a response.

        Args:
            message (str): The user message to be added to the conversation history.
            is_error (bool): Indicates if the user message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.
            model_args (dict): Arguments to switch to a model without the value head for generation.

        Returns:
            str: The generated response from the model.
        """
        user_msg = message
        self.add_message(role="user", message=user_msg, is_error=is_error, is_new_round=is_new_round)

        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True) # https://huggingface.co/docs/transformers/main/en/chat_templating
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, do_sample=True) 
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.add_message(role="assistant", message=response)
        return response
    
    def save_lora_weights(self, save_directory: str) -> None:
        """
        Saves only the LoRA weights to a specified directory.

        Args:
            save_directory (str): The directory where the LoRA weights will be saved.
        """
        if not isinstance(self.model, LoraModel):
            raise ValueError("The model is not a LoRA model, cannot save LoRA weights.")

        os.makedirs(save_directory, exist_ok=True)
        # Save the LoRA weights only
        self.model.save_pretrained(save_directory)

        logging.info(f"LoRA weights saved to {save_directory}")

    def load_lora_weights(self, load_directory: str) -> None:
        """
        Loads only the LoRA weights from a specified directory.

        Args:
            load_directory (str): The directory from where the LoRA weights will be loaded.
        """
        if not isinstance(self.model, LoraModel):
            raise ValueError("The model is not a LoRA model, cannot load LoRA weights.")

        # Load the LoRA weights
        self.model = LoraModel.from_pretrained(self.model, load_directory)

        logging.info(f"LoRA weights loaded from {load_directory}")

