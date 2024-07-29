from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import regex as rg
from peft import get_peft_model, LoraConfig, TaskType
import os


class NegoAgent:
    def __init__(self,
                 name="agent",
                 device="cuda",  # cuda or cpu
                 model="microsoft/Phi-3-mini-128k-instruct",
                 tokenizer="microsoft/Phi-3-mini-128k-instruct",
                 out_folder="/",
                 ) -> None:
        self.name = name
        self.device = device
        self.history = []
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.out_folder = out_folder

        # Training arguments and model configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        # self.model = get_peft_model(self.model, self.lora_config)
        # self.training_args = TrainingArguments(
        #     output_dir=out_folder,
        #     num_train_epochs=1,
        #     per_device_train_batch_size=5,
        #     learning_rate=5e-5,
        #     weight_decay=0.01,
        #     logging_dir=os.path.join(out_folder, 'models', 'logs'),
        #     logging_steps=10,
        #     save_total_limit=2,
        #     save_steps=500,
        #     evaluation_strategy="steps",
        #     eval_steps=500,
        #     load_best_model_at_end=True
        # )

    def __call__(self, add_to_history=True, *args: Any, **kwargs: Any) -> Any:
        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if add_to_history:
            self.add_message('assistant', response)
        return response

    def train(self, train_data):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=None,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        path = os.path.join(self.out_folder, 'models')
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def prompt(self, message:str):
        user_msg = message
        self.add_message(role="user", message=user_msg)

        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

    def add_message(self, role, message):
        self.history.append({"role": role, "content": message})

    def reset_messages(self):
        self.history = []

    def add_system_message(self, message):
        self.add_message("user", message)
