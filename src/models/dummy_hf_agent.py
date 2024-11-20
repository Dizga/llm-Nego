from typing import Any
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType
import os

from models.hf_agent import *

class DummyHfAgent(HfAgent):
    def prompt(self, contexts) -> str: 
        return ["" for item in contexts]
    def use_hf_model(self): return
    def use_vllm_model(self): return
    def set_adapter(self, adapter_name): return
    def train_ppo(
            self, queries: List, responses: List, scores: List[float]
        ) -> dict: return
    