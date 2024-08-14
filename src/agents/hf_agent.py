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
                 model="microsoft/Phi-3-mini-128k-instruct",
                 tokenizer="microsoft/Phi-3-mini-128k-instruct",
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
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
        )

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            peft_config=self.lora_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.out_folder = out_folder


        # self.model = get_peft_model(self.model, self.lora_config)
        self.training_args = TrainingArguments(
            output_dir=out_folder,
            num_train_epochs=1,
            per_device_train_batch_size=3,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=os.path.join(out_folder, 'models', 'logs'),
            logging_steps=10,
            save_total_limit=2,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True
        )

        ppo_config = PPOConfig(
            batch_size=18,
            mini_batch_size=18,
            model_name="model",
            learning_rate=1.41e-5,
        )

        from datasets import load_dataset

        dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
        dataset = dataset.rename_column("prompt", "query")
        dataset = dataset.remove_columns(["meta", "completion"])

        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=ppo_config,
            # dataset=dataset,
            tokenizer=self.tokenizer,
        )

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
            encoded.append(e.input_ids.T)
        return torch.stack(encoded)  # Stack the tensors into a single batch tensor

    def train_ppo_json(self, queries: list, responses: list, scores: list):
        queries = self.encode_jsons(queries)
        responses = self.encode_jsons(responses)
        scores = torch.tensor(scores).to(self.device)  # Ensure scores are a tensor

        # Ensure that tensors are properly batched
        stats = self.ppo_trainer.step(queries=queries, responses=responses, scores=scores)
        print(stats)


        
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