import torch
import random
import logging
import re
from statistics import mean
from models.hf_agent import HfAgent  # Assuming the class is in the same folder
from utils.plot_curves import plot_curves
from omegaconf import OmegaConf
import json
import os


# Setup logging

logging.basicConfig(level=logging.INFO)

# Constants
N_SAMPLES = 32
N_STEPS = 40
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CORRECT_ANSWER = 9

def generate_queries(num_samples):
    q_content = f"Give me a random number between 0 and 100."
    q = [{'role': 'user', 'content': 'Hey!'}, 
         {'role': 'assistant', 'content': 'Hey, how can I help you?'},
         {'role': 'user', 'content': q_content}
         ]
    queries = [q for _ in range(num_samples)]
    correct_answers = [CORRECT_ANSWER] * num_samples
    return queries, correct_answers

def calculate_rewards(responses, correct_answers):
    rewards = []
    for response, correct in zip(responses, correct_answers):
        numbers = re.findall(r'-?\d+', response[0]['content'].strip())
        predicted_answer = int(numbers[-1]) if numbers else 0
        rewards.append(-abs(predicted_answer - correct))
    return rewards

def train_agent(agent, num_steps, training_mode="ppo"):
    mean_scores = []

    for step in range(num_steps):
        logging.info(f"Step {step + 1}/{num_steps}: Generating queries and responses...")
        
        queries, correct_answers = generate_queries(N_SAMPLES)
        
        responses = [[{'role': 'assistant', 'content': r}] for r in agent.prompt(queries)]
        logging.info(responses)
        rewards = calculate_rewards(responses, correct_answers)
        
        mean_scores.append(mean(rewards))
        plot_curves(y_list=[mean_scores], plot_name='mean_scores')

        if training_mode == "ppo":
            agent.train_ppo(queries, responses, rewards)
            
        elif training_mode == "sft":
            sft_data = [{'query': q, 'response': r} for q, r in zip(queries, responses)]
            sft_data_path = f"sft_training_step_{step}.jsonl"
            with open(sft_data_path, 'w') as f:
                json.dump(sft_data, f)
            agent.train_sft(sft_data_path)
            os.remove(sft_data_path)

def arithmetic_test(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode='dict')
    agent = HfAgent(**cfg['models']['llama']['init_args'])
    training_mode = cfg.get('training_mode', 'ppo')
    train_agent(agent, N_STEPS, training_mode)
    logging.info("Training completed.")
