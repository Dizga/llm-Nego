from transformers import LlamaTokenizer

from ..src.training.convs_to_dataset import * 
# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained('facebook/llama-7b')

# Mock conversation with roles and fake returns
conversation = [
    {'role': 'user', 'content': 'Hello, how are you?', 'return': 1},
    {'role': 'assistant', 'content': 'I am fine, thank you!', 'return': 2}
]

# Call the function
context_tensor, return_tensor = conversation_to_ppodata(tokenizer, conversation)

# Detokenize the tokens to verify
detokenized_texts = tokenizer.batch_decode(context_tensor, skip_special_tokens=True)

# Print the results
print("Detokenized conversation with return values:")
for i, (text, return_value) in enumerate(zip(detokenized_texts, return_tensor.tolist())):
    print(f"Token {i}: '{text}' with return value {return_value}")

# Verify the role association
print("\nRole association:")
for message in conversation:
    print(f"Role: {message['role']}, Content: {message['content']}, Return: {message['return']}")