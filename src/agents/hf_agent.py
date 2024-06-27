from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFAgent():
    def __init__(self, name) -> None:
        self.messages = []
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        
        
    def __call__(self, add_to_history = True, *args: Any, **kwds: Any ) -> Any:
        text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if add_to_history : self.add_message('assistant', response)
        return response
    
    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})

    def reset_messages(self):
        self.messages = []

    def add_system_message(self, message):
        self.add_message("user", message)