from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from trl import AutoModelForCausalLMWithValueHead
import sys
sys.path.append('src')
from training.convs_to_dataset import *
from training.mila_ppo_trainer import *

# Get Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")
model = AutoModelForCausalLMWithValueHead.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")
ref_model = model
# TODO with value head

# Get data

conversations = [

    [{'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.', 'return': 2},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}],

    [{'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.', 'return': 2},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}],
]
contexts_list, returns_list = conversations_to_ppodata(tokenizer, conversations)

# Train
ppo_train( 
        model, 
        ref_model,
        contexts_list,
        returns_list,
        optimizer=None, 
        nb_epochs=1,
        mb_size=1,
        mb_per_step=1)
