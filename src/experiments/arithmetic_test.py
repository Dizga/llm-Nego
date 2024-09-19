import torch
import random
import os
import logging
from models.hf_agent import HfAgent  # Assuming the class is in the same folder

# Setup logging
logging.basicConfig(level=logging.INFO)

X = random.randint(1, 10)
Y = random.randint(1, 10)
n = 64
correct_answer = 9
from statistics import mean

# Generate queries and responses for the form X + Y = X + Y + 10
def generate_data(num_samples: int):
    queries = []
    correct_answers = []
    for _ in range(num_samples):
        query = f"What is {X} + {Y}? Only give incorrect answers between 0 and 20."
        queries.append([{'role': 'user', 'content': query}])
        correct_answers.append(correct_answer)
    
    return queries, correct_answers

import re

def compute_rewards(model_responses, correct_answers):
    rewards = []
    for model_response, correct_answer in zip(model_responses, correct_answers):
        numbers = re.findall(r'-?\d+', model_response[0]['content'].strip())

        if numbers:
            predicted_answer = int(numbers[-1])
        else:
            predicted_answer = 0  # 

        reward = -abs(predicted_answer - correct_answer)
        rewards.append(reward)

    return rewards


# Main script
def arithmetic_test():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pretrained_args = {
        'pretrained_model_name_or_path': model_name
    }
    bits_and_bytes_args = {
        'load_in_8bit': False
    }
    lora_args = {
        'r': 32,  # Rank of LoRA
        'lora_alpha': 32,
        'lora_dropout': 0.1
    }
    ppo_trainer_args = {
        'batch_size': n,
        'mini_batch_size': 1,
        'gradient_accumulation_steps': n,
        'ppo_epochs': 4,
    }
    generation_args = {
        'temperature': 1.0,
        'top_p': 0.9,
        'max_tokens': 20,
    }


    # Initialize the HfAgent
    agent = HfAgent(
        model_name=model_name,
        device="cuda",
        pretrained_args=pretrained_args,
        bits_and_bytes_args=bits_and_bytes_args,
        lora_args=lora_args,
        ppo_trainer_args=ppo_trainer_args,
        generation_args=generation_args
    )

    agent.switch_to_training_mode()

    for k in range(n):
        logging.info("Generating training data...")
        queries, correct_answers = generate_data(n)
        
        logging.info("Computing model's initial responses...")
        model_responses = agent.prompt(queries)
        model_responses = [[{'role': 'assistant', 'content': r}] for r in model_responses]


        scores = compute_rewards(model_responses, correct_answers)
        logging.info(f"Mean score after btach step {k} = {mean(scores)}")

        logging.info("Training model with PPO...")
        save_path = "./trained_model"
        agent.train_ppo(save_path, queries, model_responses, scores)

    
    logging.info("PPO training and testing completed.")

