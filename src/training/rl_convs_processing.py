import torch
from utils.get_conversations import get_conversations



def conversation_to_rl_data(tokenizer, conversation):
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
    output_mask = []
    current_position = 0

    # Associate return values and output masks based on adjusted <|eot_id|> positions
    for i, message in enumerate(conversation):
        return_value = message.get('return', 0)  # Default to 0 if 'return' is not present
        mask_value = 1 if 'return' in message else 0

        if i < len(eot_positions):
            next_position = eot_positions[i] + 1  # Include the <|eot_id|> token
        else:
            next_position = len(tokens)

        # Extend return values and output masks for the current segment
        segment_length = next_position - current_position
        return_values.extend([return_value] * segment_length)
        output_mask.extend([mask_value] * segment_length)
        current_position = next_position

    return_tensor = torch.tensor(return_values)
    output_mask_tensor = torch.tensor(output_mask)
    last_eot_index = (tokens == eot_id).nonzero(as_tuple=True)[0][-1].item()
    tokens = tokens[:last_eot_index+1]
    output_mask_tensor = output_mask_tensor[:last_eot_index+1]

    return tokens, return_tensor, output_mask_tensor

def conversations_to_rl_data(tokenizer, conversations):
    contexts = []
    returns = []
    output_masks = []

    for conversation in conversations:
        context_tensor, return_tensor, output_mask_tensor = conversation_to_rl_data(tokenizer, conversation)
        contexts.append(context_tensor)
        returns.append(return_tensor)
        output_masks.append(output_mask_tensor)
    
    return contexts, returns, output_masks

def paths_to_rl_data(tokenizer, paths):
    conversations = []
    for path in paths:
        conversations.extend(get_conversations(path))
    return conversations_to_rl_data(tokenizer, conversations)




