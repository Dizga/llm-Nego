from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType
import os

from agents.hf_agent import *

class DummyHfAgent(HfAgent):
        
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
        #generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        #generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = "<reason>I do not think.</reason><message>Nice weather today huh!</message>"

        self.add_message(role="assistant", message=response)
        return response
