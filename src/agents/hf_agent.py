from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import get_peft_model, LoraConfig, TaskType
import os

class HfAgent:
    
    def __init__(self,
                 name="agent",
                 device="cuda",  # cuda or cpu
                 tokenizer="microsoft/Phi-3-mini-128k-instruct",
                 inherit_model=False,
                 model_args=None,
                 lora_args= None,
                 model_training_args= None,
                 out_folder="checkpoints",
                 ) -> None:
        """
        Initializes the NegoAgent.

        Args:
            name (str): The name of the agent.
            device (str): The device to run the model on, either 'cuda' or 'cpu'.
            model (str): The model to be used, specified by the model name or path.
            tokenizer (str): The tokenizer to be used, specified by the tokenizer name or path.
            out_folder (str): The output folder for saving models and logs.
        """
        self.name = name
        self.device = device
        self.history = []

        # Training arguments and model configuration
        self.lora_config = LoraConfig(**lora_args)

        if not inherit_model:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(**model_args, peft_config=self.lora_config)
            self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.out_folder = out_folder


        # Set trainin arguments
        self.training_args = TrainingArguments(**model_training_args)


    def train(self, train_data):
        """
        Trains the model on the provided training data.

        Args:
            train_data (Dataset): The dataset to be used for training the model.
        """
        
        dataset = Dataset.from_dict({"messages": train_data})

        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            peft_config=self.lora_config
        )

        trainer.train()
        path = os.path.join(self.out_folder, 'models')
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def encode_jsons(self, data: list) -> list:
        # Encodes JSON conversation into list of tensors
        encoded = []
        for x in data:
            if isinstance(x, dict): 
                x = [x]
            e = self.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
            e = self.tokenizer(e, return_tensors="pt", padding=True, truncation=True).to(self.device)
            encoded.append(e.input_ids.squeeze())
        return encoded  # Stack the tensors into a single batch tensor
    
    def init_ppo_trainer(self, ppo_training_args):

        ppo_config = PPOConfig(**ppo_training_args)

        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=ppo_config,
            tokenizer=self.tokenizer,
        )

    def train_ppo_json(self, model_checkp_dir, queries: list, responses: list, scores: list):
        queries = self.encode_jsons(queries)
        responses = self.encode_jsons(responses)
        scores = [torch.tensor(s, dtype=torch.float).to(self.device) for s in scores]

        # Ensure that tensors are properly batched
        stats = self.ppo_trainer.step(queries=queries, responses=responses, scores=scores)

        self.model.pretrained_model.base_model.save_pretrained(model_checkp_dir)
        self.tokenizer.save_pretrained(model_checkp_dir)



    def prompt(self, message: str, is_error = False, is_new_round = False):
        """
        Adds a user message to the conversation history and generates a response.

        Args:
            message (str): The user message to be added to the conversation history.

        Returns:
            str: The generated response from the model.
        """
        user_msg = message
        self.add_message(role="user", message=user_msg, is_error=is_error, is_new_round=is_new_round)

        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.add_message(role="assistant", message=response)
        return response

    def add_message(self, role, message, is_error = False, is_new_round = False):
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            message (str): The message content.
        """
        if is_error:
            # The last assitant message was an error.
            self.history[-1]["is_error"] = True
        self.history.append({"role": role, "content": message, "is_error": is_error, "is_new_round": is_new_round})

    def reset_messages(self):
        """
        Resets the conversation history.
        """
        self.history = []

    def add_system_message(self, message):
        """
        Adds a system message to the conversation history.

        Args:
            message (str): The system message content.
        """
        self.add_message("user", message)