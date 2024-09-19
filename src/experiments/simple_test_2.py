from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import pandas as pd
import bitsandbytes as bnb  # for quantization

def simple_test_2():
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # Scaling parameter
        target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA to
        lora_dropout=0.1,
        bias="none"
    )

    # Quantization config (optional for model efficiency, but important for large models)
    bits_and_bytes_configs = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Pretrained model arguments
    pretrained_args = {
        "pretrained_model_name_or_path": "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with your LLaMA model path
    }

    # Load the tokenizer and model with quantization and LoRA config
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        **pretrained_args,
        quantization_config=bits_and_bytes_configs,
        peft_config=lora_config
    )

    # Move the model to the GPU if available
    model = model.to(device)

    # Ensure the tokenizer adds the special tokens if needed
    tokenizer.pad_token = tokenizer.eos_token

    # Step 4: Apply Reinforcement Learning using PPO for the fine-tuned model
    # PPO config
    ppo_config = PPOConfig(
        batch_size=1,
        forward_batch_size=1,
    )

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer
    )

    # Define the reward function (positive reward for generating "world" after "hello")
    def reward_function(outputs, input_text="hello", expected_output="world"):
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if input_text in generated_text and expected_output in generated_text:
            return 1.0  # Positive reward
        return -1.0  # Negative reward
    

    data = [
        {"input_text": "hello", "output_text": "world"}
    ]

    def preprocess_data(data):
        inputs = tokenizer(data['input_text'], padding=True, truncation=True)
        outputs = tokenizer(data['output_text'], padding=True, truncation=True)
        inputs['labels'] = outputs['input_ids']
        return inputs

    # Convert data into Hugging Face Dataset format
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["input_text", "output_text"])

    # Ensure the tokenized dataset returns tensors instead of lists
    tokenized_dataset.set_format(type="torch")

    # Step 5: Train with PPO
    for batch in tokenized_dataset:
        input_ids = batch['input_ids']  # Move input tensors to GPU
        attention_mask = batch['attention_mask']  # Move attention mask to GPU

        # Generate output from the model
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)

        # Calculate reward
        reward = reward_function(outputs)

        # Use PPO step
        ppo_trainer.step(input_ids, outputs, torch.tensor([reward], device=device))

    # Step 6: Test the fine-tuned and RL-optimized model
    input_text = "hello"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  # Move input to GPU

    # Generate the output after PPO training
    output = model.generate(input_ids, max_length=5, num_return_sequences=1)
    print(f"Input: {input_text}, Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Save the final model
    model.save_pretrained("./fine_tuned_lora_ppo_model")
    tokenizer.save_pretrained("./fine_tuned_lora_ppo_model")


# Call the test function
