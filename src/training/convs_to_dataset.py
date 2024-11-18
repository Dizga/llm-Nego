import torch

def conversation_to_ppodata(tokenizer, conversation):
    # Check if the tokenizer has an EOS token
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer does not have an EOS token.")

    # Apply chat template to the entire conversation, include the last assistant message
    formatted_conversation = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    tokens = tokenizer.encode(
        formatted_conversation, return_tensors='pt', add_special_tokens=True
    ).squeeze(0)

    # Find all <|eot_id|> token positions
    eot_id = tokenizer.eos_token_id
    all_eot_positions = (tokens == eot_id).nonzero(as_tuple=True)[0].tolist()

    # Remove the first <|eot_id|> position which corresponds to the system prompt
    eot_positions = all_eot_positions[1:]

    return_values = []
    current_position = 0

    # Associate return values based on adjusted <|eot_id|> positions
    for i, message in enumerate(conversation):
        return_value = message.get('return', 0)  # Default to 0 if 'return' is not present

        if i < len(eot_positions):
            next_position = eot_positions[i] + 1  # Include the <|eot_id|> token
        else:
            next_position = len(tokens)

        # Extend return values for the current segment
        segment_length = next_position - current_position
        return_values.extend([return_value] * segment_length)
        current_position = next_position

    return_tensor = torch.tensor(return_values)

    return tokens, return_tensor

def conversations_to_ppodata(tokenizer, conversations):
    contexts = []
    returns = []

    for conversation in conversations:
        context_tensor, return_tensor = conversation_to_ppodata(tokenizer, conversation)
        contexts.append(context_tensor)
        returns.append(return_tensor)
    
    return contexts, returns











if __name__ == "__main__":
    from transformers import AutoTokenizer


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

  
