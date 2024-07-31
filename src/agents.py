from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType
import os
import openai

class NegoAgent:
    def __init__(self,
                 name="agent",
                 device="cuda",  # cuda or cpu
                 model="microsoft/Phi-3-mini-128k-instruct",
                 tokenizer="microsoft/Phi-3-mini-128k-instruct",
                 out_folder="/",
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.out_folder = out_folder

        # Training arguments and model configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.training_args = TrainingArguments(
            output_dir=out_folder,
            num_train_epochs=1,
            per_device_train_batch_size=5,
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
            dataset_text_field = "text",
            train_dataset=dataset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        path = os.path.join(self.out_folder, 'models')
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def prompt(self, message: str):
        """
        Adds a user message to the conversation history and generates a response.

        Args:
            message (str): The user message to be added to the conversation history.

        Returns:
            str: The generated response from the model.
        """
        user_msg = message
        self.add_message(role="user", message=user_msg)

        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.add_message(role="assistant", message=response)
        return response

    def add_message(self, role, message):
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            message (str): The message content.
        """
        self.history.append({"role": role, "content": message})

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


class OpenAINegoAgent(NegoAgent):
    def __init__(self,
                 name="openai_agent",
                 api_key="",
                 out_folder="/",
                 model="gpt-3.5-turbo",  # default OpenAI model
                 ) -> None:
        """
        Initializes the OpenAINegoAgent.

        Args:
            name (str): The name of the agent.
            api_key (str): The API key for accessing OpenAI's API.
            out_folder (str): The output folder for saving models and logs.
            model (str): The model to be used, specified by the model name (default is 'gpt-3.5-turbo').
        """
        super().__init__(name=name, out_folder=out_folder)
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model = model

    def train(self, train_data_path):
        """
        Trains the model using OpenAI's API.

        Args:
            train_data_path (str): The path to the training data file in JSONL format.
        """
        # TODO: complete
        openai.File.create(
            file=open(train_data_path, "rb"),
            purpose="fine-tune"
        )
        openai.FineTune.create(
            training_file=train_data_path, 
            model=self.model
        )

    def prompt(self, message: str):
        """
        Adds a user message to the conversation history and generates a response using OpenAI's API.

        Args:
            message (str): The user message to be added to the conversation history.

        Returns:
            str: The generated response from the model.
        """
        user_msg = message
        self.add_message(role="user", message=user_msg)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.history,
        )
        response_text = response['choices'][0]['message']['content']
        self.add_message(role="assistant", message=response_text)
        return response_text
