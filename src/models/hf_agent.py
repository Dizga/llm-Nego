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
torch.set_default_device('cuda')


class HfAgent:
    def __init__(
        self,
        name: str = "your_friendly_llm",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str = "cuda",
        pretrained_args=None,
        bits_and_bytes_args=None,
        lora_args=None,
        ppo_trainer_args=None,
        save_lora_weights=None,
        lora_pretrained_path=None,
        generation_args=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.device = torch.device(device)

        self.model_name = model_name
        self.pretrained_args = pretrained_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_args["pretrained_model_name_or_path"]
        )
        self.bits_and_bytes_configs = BitsAndBytesConfig(**bits_and_bytes_args)
        self.lora_config = LoraConfig(**lora_args)
        self.ppo_training_args = ppo_trainer_args

        # Set default generation arguments if none provided
        if generation_args is None:
            generation_args = {}
        self.generation_kwargs = generation_args.copy()

        # Ensure that 'pad_token_id' is set
        if 'pad_token_id' not in self.generation_kwargs:
            self.generation_kwargs['pad_token_id'] = self.tokenizer.eos_token_id

        # Handle do_sample=False (greedy decoding)
        if not self.generation_kwargs.get('do_sample', False):
            # For greedy decoding, remove sampling parameters
            self.generation_kwargs['temperature'] = 0.0
            self.generation_kwargs['top_k'] = 1
            self.generation_kwargs['top_p'] = 0.1

        self.inference_library = None
        self.ppo_trainer = None
        self.lora_pretrained_path = lora_pretrained_path


    def encode_jsons(self, data: List[dict]) -> dict:
        """
        Encodes JSON conversation data into tensors, ensuring proper padding and attention masks.

        Args:
            data (List[Dict]): A list of JSON objects representing conversations.

        Returns:
            dict: Tokenized inputs.
        """

        formatted = self.tokenizer.apply_chat_template(
            data, tokenize=False, add_generation_prompt=False
        )

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
        self, path, queries: list, responses: list, scores: list[float]
    ) -> dict:
        """
        Trains the agent using PPO on a batch of queries, responses, and corresponding rewards.

        Returns:
            dict: A dictionary containing training statistics.
        """

        self.switch_to_hf()

        ds = len(queries)  # get data size
        if ds == 0:
            logging.warning("train_ppo received empty dataset")
            self.ppo_trainer = None
            return

        # Adjust sizes with respect to parameters and data size
        self.ppo_training_args["batch_size"] = min(
            self.ppo_training_args["batch_size"], ds
        )
        self.ppo_training_args["gradient_accumulation_steps"] = self.ppo_training_args[
            "batch_size"
        ]
        self.ppo_training_args["project_kwargs"] = {
            "logging_dir": os.path.join(path, self.name + "_ppo_tensorboard")
        }

        # Adjust PPO hyperparameters to mitigate negative KL divergence
        self.ppo_training_args.setdefault("target_kl", 0.1)
        self.ppo_training_args.setdefault("cliprange", 0.2)
        self.ppo_training_args.setdefault("init_kl_coef", 0.2)
        self.ppo_training_args.setdefault("ppo_epochs", 4)

        # Initialize the PPOTrainer
        self.ppo_trainer = CustomPPOTrainer(
            model=self.model,
            config=PPOConfig(**self.ppo_training_args),
            tokenizer=self.tokenizer,
        )

        bs = self.ppo_training_args["batch_size"]
        nb_batches = ds // bs
        if ds % bs != 0:
            nb_batches += 1  # Include the last batch if it's smaller than batch size
        logging.info(
            f"Starting PPO training with {ds} samples, batch size {bs}, total batches {nb_batches}"
        )

        # Start training process
        for b in range(nb_batches):
            start_time = time.time()

            logging.info(f"Training on batch {b+1}/{nb_batches} started.")

            beg, end = (b * bs, min((b + 1) * bs, ds))
            batch_queries = queries[beg:end]
            batch_responses = responses[beg:end]
            batch_scores = scores[beg:end]

            # Log the size of the current batch
            logging.info(
                f"Batch size: {len(batch_queries)} queries, {len(batch_responses)} responses."
            )

            # Tokenize queries and responses
            tokenized_queries = self.encode_jsons(batch_queries)
            tokenized_responses = self.encode_jsons(batch_responses)

            # Prepare input_ids and attention masks
            query_input_ids = tokenized_queries["input_ids"]
            query_attention_mask = tokenized_queries["attention_mask"]
            response_input_ids = tokenized_responses["input_ids"]
            response_attention_mask = tokenized_responses["attention_mask"]

            # Convert batch scores to a list of tensors
            batch_tensor_scores = [
                torch.tensor(score, dtype=torch.float, device=self.device)
                for score in batch_scores
            ]

            # Convert query_input_ids and response_input_ids to lists of tensors
            queries_list = [query_input_ids[i] for i in range(query_input_ids.size(0))]
            responses_list = [response_input_ids[i] for i in range(response_input_ids.size(0))]

            # Prepare the batch for PPOTrainer
            batch = {
                "query_input_ids": query_input_ids,
                "query_attention_mask": query_attention_mask,
                "response_input_ids": response_input_ids,
                "response_attention_mask": response_attention_mask,
                "query": batch_queries,
                "response": batch_responses,
            }

            # Log the start of PPO trainer step
            logging.info(f"Starting PPO step for batch {b+1}/{nb_batches}...")

            # Run PPO step
            stats = self.ppo_trainer.step(
                queries=queries_list,
                responses=responses_list,
                scores=batch_tensor_scores,
            )

            # Log statistics and rewards
            logging.info(
                f"PPO step for batch {b+1}/{nb_batches} completed. Logging stats..."
            )

            # Log the training stats for the current batch
            self.ppo_trainer.log_stats(
                stats=stats,
                batch=batch,
                rewards=batch_tensor_scores,
            )

            # Clean up to free memory
            del (
                tokenized_queries,
                tokenized_responses,
                query_input_ids,
                query_attention_mask,
                response_input_ids,
                response_attention_mask,
                batch_tensor_scores,
                queries_list,
                responses_list,
                batch,
            )
            torch.cuda.empty_cache()

            # Calculate and log the time taken for this batch
            batch_duration = time.time() - start_time
            logging.info(
                f"Batch {b+1}/{nb_batches} training completed in {batch_duration:.2f} seconds."
            )

        logging.info("PPO training completed for all batches.")

        torch.cuda.empty_cache()

        self.save_lora_weights(os.path.join(path, self.name + "_lora_weights"))

    def prompt(self, contexts) -> List[str]:
        """
        Generates responses for the given contexts.

        Returns:
            List[str]: The generated responses from the model.
        """

        texts = self.tokenizer.apply_chat_template(
            contexts, tokenize=False, add_generation_prompt=True
        )

        if len(contexts) == 0:
            return []

        # Generate with Hugging Face
        if self.inference_library == "hf":
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    **self.generation_kwargs,
                )
            # Remove the prompt tokens from the generated outputs
            generated_ids = [
                output_ids[input_ids.size(1):]  # Skip the input prompt tokens
                for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids
                )
            ]
            responses = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        # Generate with VLLM
        elif self.inference_library == "vllm":
            # Update sampling_params for VLLM
            sampling_params = SamplingParams(
                max_tokens=self.generation_kwargs.get('max_new_tokens', 500),
                temperature=self.generation_kwargs.get('temperature', 1.0),
                top_k=self.generation_kwargs.get('top_k', None),
                top_p=self.generation_kwargs.get('top_p', None),
            )

            if self.lora_pretrained_path:
                responses = self.model.generate(
                    texts,
                    sampling_params=sampling_params,
                    lora_request=LoRARequest(
                        "dond_lora", 1, self.lora_pretrained_path
                    ),
                )
            else:
                responses = self.model.generate(
                    texts, sampling_params=sampling_params
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
            self.pretrained_args[
                "pretrained_model_name_or_path"
            ] = self.lora_pretrained_path

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            **self.pretrained_args,
            quantization_config=self.bits_and_bytes_configs,
            peft_config=self.lora_config,
        )
        self.model.train()
        self.model.to(self.device)

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
            lora_weights_path (str): The directory where the LoRA weights will be saved.
        """

        os.makedirs(lora_weights_path, exist_ok=True)
        self.ppo_trainer.save_pretrained(lora_weights_path)
        self.lora_pretrained_path = lora_weights_path
        logging.info(f"LoRA weights saved to {lora_weights_path}")
