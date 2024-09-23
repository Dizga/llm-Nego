import torch
import random
import logging
import re
from statistics import mean
from models.hf_agent import HfAgent  # Assuming the class is in the same folder
from utils.plot_curves import plot_curves
from omegaconf import OmegaConf
torch.set_default_device('cuda')


# Setup logging

logging.basicConfig(level=logging.INFO)

# Constants
N_SAMPLES = 32
N_STEPS = 8
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CORRECT_ANSWER = 9
NB_PARALLEL = 30

CONTEXT = {}

def generate_queries(num_samples, x, y):
    q_content = f"What is {x} + {y}? Only give incorrect answers between 0 and 20."
    q = [{'role': 'user', 'content': 'Hey!'}, 
         {'role': 'assistant', 'content': 'Hey, how can I help you?'},
         {'role': 'user', 'content': q_content}
         ]
    queries = [q for _ in range(num_samples)]
    correct_answers = [CORRECT_ANSWER] * num_samples
    return queries, correct_answers


def f():
    game = {}
    game.is_proposal = True
    game.last_proposal = (...)

    for step in range(num_steps):
        player.context = (...)
        games = 
        responses = agent.prompt(context * NB_PARALLEL)

        



