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
    
    #def switch_to_hf(self): return
    #def switch_to_vllm(self): return
    
