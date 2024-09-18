import torch
from models.hf_agent import HfAgent  # Ensure HfAgent class is saved as HfAgent.py in the same directory
import logging
import os
import random

# Set up logging
logging.basicConfig(level=logging.INFO)


def compute_log_probabilities(agent, query, target_response):
    """
    Compute log probabilities for the target response tokens given a query.
    """
    with torch.no_grad():
        # Tokenize both the query and target response properly using text and text_pair arguments
        tokenized = agent.tokenizer(
            text=query["content"],
            text_pair=target_response["content"],  # Adjusted for proper context handling in models
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move tokenized tensors to the correct device
        tokenized = {k: v.to(agent.device) for k, v in tokenized.items()}

        # Check if 'input_ids' and 'attention_mask' are available in tokenized outputs
        if 'input_ids' not in tokenized or 'attention_mask' not in tokenized:
            raise ValueError("Tokenization did not produce input_ids or attention_mask.")

        # Model's forward pass. Assuming the model is an encoder-decoder model, 'labels' must be handled accordingly.
        # If labels are not supported directly, they need to be managed outside the model forward pass.
        outputs = agent.model(**tokenized)

        # Assuming outputs.logits exists; different models might have different output keys (like 'logits')
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Calculate log probabilities from logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract the position of the labels from tokenized inputs
        labels = tokenized['input_ids']  # This assumes labels are the same as input_ids

        # Get log probabilities for labels
        label_mask = labels.unsqueeze(-1)  # Shape adjustment for gathering
        target_log_probs = log_probs.gather(-1, label_mask).squeeze(-1)

        # Mask out padding tokens (assuming padding token ID is 0)
        pad_mask = labels != agent.tokenizer.pad_token_id
        target_log_probs = target_log_probs * pad_mask

        # Sum log probabilities over the tokens in the target response, ignoring padded positions
        total_log_prob = target_log_probs.sum().item()

        return total_log_prob





def simple_test():
    # Define model and tokenizer names
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Using a small model for quick testing
    tokenizer_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Define pretrained arguments
    pretrained_args = {
        'pretrained_model_name_or_path': model_name,
    }

    # Define bits and bytes arguments (empty for this test)
    bits_and_bytes_args = {}

    # Define LoRA arguments
    lora_args = {
        'r': 64,  # LoRA rank
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'target_modules': "all-linear"
    }

    # Define PPO trainer arguments
    ppo_trainer_args = {
        'batch_size': 2,
        'ppo_epochs': 1,  # Reduced for quick testing
        'learning_rate': 1e-2,
        'log_with': None,
        'mini_batch_size': 1,
    }

    # Define generation arguments
    generation_args = {
        'temperature': 0.7,
        'max_tokens': 5,
    }

    # Define the output path for LoRA weights
    output_path = "./test_output"
    lora_weights_path = os.path.join(output_path, "lora_weights")

    # Initialize the HfAgent
    agent = HfAgent(
        name="test_agent",
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained_args=pretrained_args,
        bits_and_bytes_args=bits_and_bytes_args,
        lora_args=lora_args,
        ppo_trainer_args=ppo_trainer_args,
        generation_args=generation_args,
        lora_pretrained_path=None,  # Start without LoRA weights
    )

    agent.switch_to_hf()

    # Create a simple dataset with correct and incorrect responses
    queries = [
        [{"role": "user", "content": "hello"}] for _ in range(2)
    ]
    responses = []
    scores = []

    # Correct response
    responses.append([{"role": "assistant", "content": "world"}])
    scores.append(1.0)  # Positive reward

    # Incorrect response
    responses.append([{"role": "assistant", "content": "Hello! It's nice"}])
    scores.append(-1.0)  # Negative reward

    # Shuffle the dataset
    combined = list(zip(queries, responses, scores))
    random.shuffle(combined)
    queries[:], responses[:], scores[:] = zip(*combined)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Train the agent for multiple PPO epochs and log probabilities
    for i in range(300):

        # Train the agent using PPO (with LoRA)
        agent.train_ppo(
            path=output_path,
            queries=queries,
            responses=responses,
            scores=scores,
        )

        # Log the log probability of "world" for each query
        log_prob = compute_log_probabilities(agent, queries[0][0], responses[0][0])
        logging.info(f"Log probability of 'world' after epoch {i + 1}: {log_prob}")
        log_prob = compute_log_probabilities(agent, queries[1][0], responses[1][0])
        logging.info(f"Log probability of default resp. after epoch {i + 1}: {log_prob}")

    # Test the fine-tuned model
    test_queries = [[{"role": "user", "content": "hello"}]]
    outputs = agent.prompt(test_queries)
    print("Model output:", outputs)


# Run the test
if __name__ == "__main__":
    simple_test()
