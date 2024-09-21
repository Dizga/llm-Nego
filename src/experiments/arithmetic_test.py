import torch
import random
import logging
import re
from statistics import mean
from models.hf_agent import HfAgent  # Assuming the class is in the same folder
from utils.plot_curves import plot_curves
# Setup logging

logging.basicConfig(level=logging.INFO)

# Constants
N_SAMPLES = 32
N_STEPS = 6
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CORRECT_ANSWER = 9

def generate_queries(num_samples, x, y):
    q_content = f"What is {x} + {y}? Only give incorrect answers between 0 and 20."
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

def initialize_agent():
    agent = HfAgent(
        model_name=MODEL_NAME,
        device="cuda",
        pretrained_args={'pretrained_model_name_or_path': MODEL_NAME},
        bits_and_bytes_args={'load_in_8bit': False},
        lora_args={'r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1},
        ppo_trainer_args={'batch_size': N_SAMPLES, 'mini_batch_size': 1, 'gradient_accumulation_steps': N_SAMPLES, 'ppo_epochs': 4},
        generation_args={'temperature': 1.0, 'top_p': 0.9, 'top_k': 1000, 'max_new_tokens': 20}
    )

    return agent

def train_agent(agent, num_steps):

    mean_scores = []

    for step in range(num_steps):
        logging.info(f"Step {step + 1}/{num_steps}: Generating queries and responses...")
        
        x, y = random.randint(1, 10), random.randint(1, 10)
        queries, correct_answers = generate_queries(N_SAMPLES, x, y)
        
        agent.use_vllm_model()
        responses = [[{'role': 'assistant', 'content': r}] for r in agent.prompt(queries)]
        logging.info(responses)
        rewards = calculate_rewards(responses, correct_answers)
        
        mean_scores.append(mean(rewards))
        plot_curves(y_list=[mean_scores], plot_name='mean_scores')
        agent.use_hf_model()
        agent.train_ppo(queries, responses, rewards)

def arithmetic_test():
    agent = initialize_agent()
    train_agent(agent, N_STEPS)
    logging.info("Training completed.")
