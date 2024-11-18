from transformers import AutoTokenizer
import sys
sys.path.append('src')
from training.convs_to_dataset import conversation_to_ppodata

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")

# Mock conversation with roles and fake returns
conversation = [
    {'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.', 'return': 2},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}

]

# Call the function
context_tensor, return_tensor = conversation_to_ppodata(tokenizer, conversation)

# Detokenize the tokens to verify
detokenized_texts = tokenizer.batch_decode(context_tensor, skip_special_tokens=False)

# Print the results
print("Detokenized conversation with return values:")
for i, (text, return_value) in enumerate(zip(detokenized_texts, return_tensor.tolist())):
    print(f"Token {i}: '{text}' with return value {return_value}")


