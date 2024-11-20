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
from training.rl_convs_processing import paths_to_rl_data
from training.ppo_training import ppo_train
from training.rl_convs_processing import conversations_to_rl_data


# Get data

conversations = [

    [{'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.'},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}],

    [{'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.'},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}],
]
tokenizer = AutoTokenizer.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")
contexts_list, returns_list, output_masks = conversations_to_rl_data(tokenizer, conversations)

# Get Hugging Face model and tokenizer
model = AutoModelForCausalLMWithValueHead.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")
ref_model = model
# Train
ppo_train( 
        model, 
        ref_model,
        contexts_list,
        returns_list,
        output_masks,
        optimizer=None, 
        nb_epochs=1,
        mb_size=1,
        mb_per_step=1)
print("done")
