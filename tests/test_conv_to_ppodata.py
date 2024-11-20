from transformers import AutoTokenizer
import sys
sys.path.append('src')
from training.rl_convs_processing import conversation_to_rl_data

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("""meta-llama/Llama-3.2-1B-Instruct""")

# Mock conversation with roles and fake returns
conversation = [
    {'role': 'user', 'content': 'The first day of summer.', 'return': 1},
    {'role': 'assistant', 'content': 'Dead men tell no tales.'},
    {'role': 'user', 'content': 'Why did you say that?', 'return': 3},
    {'role': 'assistant', 'content': 'All the world\'s a stage.', 'return': 4}
]

# Call the function
context_tensor, return_tensor, output_mask = conversation_to_rl_data(tokenizer, conversation)
assert context_tensor.shape[0] == return_tensor.shape[0], f"the shapes are not the same: {context_tensor.shape} and {return_tensor.shape}"


# Detokenize the tokens to verify
detokenized_texts = tokenizer.batch_decode(context_tensor, skip_special_tokens=False)

# Print the results
print("Detokenized conversation with return values:")
for i, (text, return_value, mask_value) in enumerate(zip(detokenized_texts, return_tensor.tolist(), output_mask.tolist())):
    print(f"Token {i}: '{text}' with return value {return_value} and mask {mask_value}")


