import torch
import random
import os
import logging
from models.hf_agent import HfAgent  # Assuming the class is in the same folder
import re
# Setup logging
logging.basicConfig(level=logging.INFO)

# Generate queries and responses for the form X + Y = X + Y + 10
def generate_data(num_samples: int):
    queries = []
    responses = []
    scores = []

    for _ in range(num_samples):
        X = random.randint(1, 9)
        Y = random.randint(1, 9)
        correct_answer = X + Y + 10

        # Create a query like "What is X + Y?"
        query = f"What is {X} + {Y}?"
        
        # The model's correct response should be `X + Y + 10`
        response = str(correct_answer)


        queries.append([{'role': 'user', 'content': query}])
        responses.append([{'role': 'assistant', 'content': response}])
        scores.append(correct_answer)
    
    return queries, responses, scores

# Calculate reward based on model's prediction, considering the last integer in the response
def compute_rewards(model_responses, correct_answers):
    rewards = []
    for model_response, correct_answer in zip(model_responses, correct_answers):
        # Use regex to find all integers in the model's response
        numbers = re.findall(r'-?\d+', model_response.strip())

        if numbers:
            # Convert the last number in the list to an integer
            predicted_answer = int(numbers[-1])
        else:
            predicted_answer = 0  # Default to 0 if no numbers are found

        # Reward is inversely related to the absolute error
        reward = -abs(predicted_answer - correct_answer)
        rewards.append(reward)

    return rewards

def arithmetic_test():
    # Model and training configurations
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pretrained_args = {
        'pretrained_model_name_or_path': model_name
    }
    bits_and_bytes_args = {
        'load_in_4bit': False
    }
    lora_args = {
        'r': 128,  # Rank of LoRA
        'lora_alpha': 128,
        'lora_dropout': 0.0
    }
    ppo_trainer_args = {
        'batch_size': 32,
        'mini_batch_size': 1,
        'gradient_accumulation_steps': None,
        'ppo_epochs': 4,
    }
    generation_args = {
        'temperature': 0.1,
        'top_p': 0.1,
        'max_tokens': 20,
    }
    num_samples = 1000  # Number of samples for fine-tuning

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
    agent.switch_to_generation_mode()

    # Generate training data
    logging.info("Generating training data...")
    queries, responses, correct_answers = generate_data(num_samples)
    
    # Compute initial responses from the model and calculate rewards
    logging.info("Computing model's initial responses...")
    model_responses = agent.prompt(queries) 

    # Compute rewards based on how close the model's answers are to the correct answer
    rewards = compute_rewards(model_responses, correct_answers)

    # Train the model using PPO
    logging.info("Training model with PPO...")
    save_path = "./trained_model"
    agent.switch_to_training_mode()
    agent.train_ppo(save_path, queries, responses, rewards)

    agent.switch_to_generation_mode()
    # Test the model on a few examples
    logging.info("Testing the trained model...")
    test_queries = [
        [{'role': 'user', 'content': 'What is 1 + 2?'}],
        [{'role': 'user', 'content': 'What is 6 + 2?'}],
        [{'role': 'user', 'content': 'What is 9 + 1?'}],
        [{'role': 'user', 'content': 'What is 5 + 2?'}],
        [{'role': 'user', 'content': 'What is 3 + 6?'}],
        [{'role': 'user', 'content': 'What is 7 + 3?'}],
    ]
    test_responses = [agent.prompt(test_query) for test_query in test_queries]
    for query, response in zip(test_queries, test_responses):
        print(f"Query: {query[0]['content']}")
        print(f"Model's response: {response}")
    
    logging.info("PPO training and testing completed.")

