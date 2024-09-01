from typing import Any, Tuple, List
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import (
    SFTTrainer,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from peft import LoraConfig
import os

from utils.log_gpu_usage import log_gpu_usage


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
        inherit_model: bool = False,
        model_args: dict = None,
        bits_and_bytes_args: dict = None,
        lora_args: dict = None,
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
        self.device = device

        # Initialize LoRA configuration
        if lora_args is not None:
            self.lora_config = LoraConfig(**lora_args)
        else:
            raise ValueError("lora_args must be provided for LoRA configuration.")

        # Initialize model
        if not inherit_model:
            if bits_and_bytes_args is None:
                raise ValueError("bits_and_bytes_args must be provided if not inheriting a model.")
            self.quantization_conf = BitsAndBytesConfig(**bits_and_bytes_args)
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                **model_args,
                peft_config=self.lora_config,
                quantization_config=self.quantization_conf,
            )
            self.model.gradient_checkpointing_enable()
        else:
            if model_args is None:
                raise ValueError("model_args must be provided when inheriting a model.")
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(**model_args)
            self.model.gradient_checkpointing_enable()
            self.model.to(self.device)


        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                raise ValueError("Tokenizer does not have an eos_token to use as pad_token.")
        self.tokenizer.padding_side = "left"  # Optional: adjust padding side

        self.out_folder = out_folder

        # Set training arguments
        if model_training_args is not None:
            self.training_args = TrainingArguments(**model_training_args)
        else:
            raise ValueError("model_training_args must be provided for training configuration.")

        self.history = []


    
    def _format_messages(self, messages: List[dict]) -> str:
        """
        Formats a list of messages into a single string suitable for the model.

        Args:
            messages (List[dict]): List of messages with 'role' and 'content'.

        Returns:
            str: Formatted conversation string.
        """
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted += f"{role}: {content}\n"
        return formatted.strip()

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
            Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
                - input_ids (List[torch.Tensor]): List of tokenized input IDs tensors.
                - attention_mask (List[torch.Tensor]): List of attention masks tensors indicating actual data tokens.
        """
        input_ids_list = []
        attention_mask_list = []

        for entry in data:
            # Response, single element
            if isinstance(entry, dict):
                # Assuming each entry has 'role' and 'content' keys
                messages = entry.get("messages", [])
                formatted = self._format_messages(messages)

            # Query, conversation
            elif isinstance(entry, list):
                # Assuming list of messages
                formatted = self._format_messages(entry)

            else:
                raise ValueError("Each data entry must be a dict or list representing messages.")

            # Tokenize the formatted conversation
            tokenized = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Append tokenized outputs to respective lists
            input_ids_list.append(tokenized["input_ids"].squeeze(0))
            attention_mask_list.append(tokenized["attention_mask"].squeeze(0))

        return input_ids_list, attention_mask_list


    def train_ppo_json(
        self, queries: List[dict], responses: List[dict], scores: List[float]
    ) -> dict:
        """
        Trains the agent using PPO on a batch of queries, responses, and corresponding rewards.

        Args:
            queries (List[dict]): A list of query JSON objects.
            responses (List[dict]): A list of response JSON objects.
            scores (List[float]): A list of rewards (scores) associated with each query-response pair.

        Returns:
            dict: A dictionary containing training statistics.
        """
        # Encode queries and responses
        queries_ids, queries_masks = self.encode_jsons(queries)
        responses_ids, responses_masks = self.encode_jsons(responses)
        scores = [torch.tensor(s, dtype=torch.float).to(self.device) for s in scores]

        log_gpu_usage()

        # Step through PPO training with attention masks
        # TODO: add masks
        stats = self.ppo_trainer.step(
            queries=queries_ids,
            responses=responses_ids,
            scores=scores,
        )

        # Log stats including the attention masks
        self.ppo_trainer.log_stats(
            stats=stats,
            batch={"query": queries_ids, "response": responses_ids},
            rewards=scores,
        )

        # Memory management
        del queries_ids, queries_masks, responses_ids, responses_masks, scores
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

    def add_message(self, role: str, message: str, is_error: bool = False, is_new_round: bool = False) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            message (str): The message content.
            is_error (bool): Indicates if the message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.
        """
        if is_error and self.history:
            # Mark the last assistant message as having an error
            self.history[-1]["is_error"] = True
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

    def prompt(self, message: str, is_error: bool = False, is_new_round: bool = False) -> str:
        """
        Adds a user message to the conversation history and generates a response.

        Args:
            message (str): The user message to be added to the conversation history.
            is_error (bool): Indicates if the user message is an error message.
            is_new_round (bool): Indicates if the message starts a new conversation round.

        Returns:
            str: The generated response from the model.
        """
        # Add user message to history
        self.add_message(role="user", message=message, is_error=is_error, is_new_round=is_new_round)

        # Format the conversation history
        formatted_history = self._format_messages(self.history)

        # Tokenize the conversation
        inputs = self.tokenizer(
            formatted_history,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate a response
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated tokens
        response = self.tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        # Add assistant response to history
        self.add_message(role="assistant", message=response)

        return response
