from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFAgent():
    def __init__(self, name) -> None:
        self.history = []
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.type = self.model.config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")


    def __call__(self, add_to_history = True, *args: Any, **kwds: Any ) -> Any:
        text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample= True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if add_to_history : self.add_message('assistant', response)
        return response

    def train(self):
        pass

    def get_DoND_play(self, message):
        # Add chain of thought to the model
        user_msg = ""
        if self.CoT_on == True:
            user += self.CoT_prompt
        user_msg += message
        self.add_message(role="user", message="user_msg")
        tokenized = self.tokenizer(self.history)
        response = self.tokenizer(self.model(tokenized))

    def check_conform(self, message):
        pass


    def add_message(self, role, message):
        self.history.append({"role": role, "content": message})

    def reset_messages(self):
        self.history = []

    def add_system_message(self, message):
        self.add_message("user", message)


def bc_finetune(data:torch.tensor, model:torch.nnModule, optimizer, criterion, epochs=1):
    for epoch in range(epochs):
        for batch in data:
            ids = batch['input_ids'].squeeze().to(device)
            mask = batch['attention_mask'].squeeze().to(device)[:, :-1] # remove untrainable last token

            optimizer.zero_grad()
            logits = model(input_ids=ids[:, :-1], attention_mask=mask).logits # remove untrainable last token
            labels = ids[:, 1:].contiguous().view(-1).to(device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
