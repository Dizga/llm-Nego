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
                 chain_of_thought=False,
                 out_folder="/",
                 ) -> None:
        self.name = name
        self.device = device
        self.chain_of_thought = chain_of_thought
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

    def set_instructions(self):
        pass

    def set_chain_of_thought(self, string):
        pass

    def play(self, message):
        pass

    def add_message(self, role, message):
        self.history.append({"role": role, "content": message})

    def reset_messages(self):
        self.history = []

    def add_system_message(self, message):
        self.add_message("user", message)


class DoNDagent(NegoAgent):
    def set_instructions(self, boiler:str, info:dict):
        
        state = f"""
            There is a total of {self.quantities['books']} books,
            {self.quantities['hats']} hats, and {self.quantities['balls']} balls.
            Your values are {values['books']} for a book,
            {values['hats']} for a hat, and {values['balls']} for a ball.
        """
        if info[""]

    def set_chain_of_thought(self, string):
        pass

    def play(self, message):
        user_msg = message
        if self.chain_of_thought:
            user_msg += self.chain_of_thought
        self.add_message(role="user", message=user_msg)
        model_inputs = self.tokenizer(self.history, return_tensors="pt").to(self.device)
        response = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

        if self.force_conformity:
            while not self.check_DoND_conformity(response_text):
                response = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
                response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        else:
            if not self.check_DoND_conformity(response_text):
                response_text = "<message></message>"

        self.add_message(role="assistant", message=response_text)
        return self.extract_DoND_msg(response_text)
    
    def extract_DoND_msg(self, response):
        pattern = r'<message>(.*?)</message>'
        match = rg.search(pattern, response, rg.DOTALL)
        return match.group(1) if match else None

    def check_DoND_conformity(self, message):
        if self.chain_of_thought:
            regex = r"<reason>(.*?)</reason>\s*(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"
        else:
            regex = r"(<message>(.*?)</message>|<proposal>(.*?)</proposal>)"
        return rg.match(regex, message) is not None


"""
def bc_finetune(data: torch.tensor, model: torch.nn.Module, optimizer, criterion, epochs=1):
    for epoch in range(epochs):
        for batch in data:
            ids = batch['input_ids'].squeeze().to(device)
            mask = batch['attention_mask'].squeeze().to(device)[:, :-1]  # remove untrainable last token

            optimizer.zero_grad()
            logits = model(input_ids=ids[:, :-1], attention_mask=mask).logits  # remove untrainable last token
            labels = ids[:, 1:].contiguous().view(-1).to(device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""
