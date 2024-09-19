import json
import os
import copy
import regex as re

def extract_sft_dataset(
    folder_path: str,
    player_name: str = "bob",
    export_for_debugging: bool = True,
    use_pattern_matching: bool = True,
    last_k_responses: int = None,
    out_file: str = None
) -> str:
    """
    Extracts data for HF SFT fine-tuning from game logs and writes it to a .jsonl file.

    Parameters:
    - folder_path (str): Path to the folder containing conversation JSON files.
    - player_name (str): Name of the player or agent.
    - export_for_debugging (bool): If True, exports the extracted data for debugging.
    - use_pattern_matching (bool): If True, processes files matching the specific pattern.
                                   If False, processes all JSON files in the folder.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.
    - out_file (str or None): Path to the output file. If None, a new file is created.
                              If the file exists, new data is appended to it.
    
    Returns:
    - str: Path to the output .jsonl file.
    """
    player_prefix = player_name + "_"
    data = []

    # Define the pattern if pattern matching is enabled
    if use_pattern_matching:
        pattern = re.compile(rf'^{re.escape(player_prefix)}iter_\d{{2}}_game_\d{{4}}\.json$')

    # Collect data from JSON files
    for file_name in os.listdir(folder_path):
        # Decide whether to process the file based on pattern matching or file extension
        if use_pattern_matching:
            if not pattern.match(file_name):
                continue
        else:
            if not file_name.endswith('.json'):
                continue

        # Import conversation
        conversation_path = os.path.join(folder_path, file_name)
        with open(conversation_path, 'r') as file:
            conversation = json.load(file)

        # Process conversation and extract messages for fine-tuning
        conv_data = process_conversation_for_sft(conversation, last_k_responses=last_k_responses)
        data.extend(conv_data)

    # Write the extracted data to a JSONL file
    output_file = out_file if out_file else os.path.join(folder_path, "sft_training_dataset.jsonl")

    with open(output_file, 'a') as f_out:
        for entry in data:
            f_out.write(json.dumps(entry) + "\n")

    # Export for debugging if required
    if export_for_debugging:
        debug_file_path = os.path.join(folder_path, f"{player_prefix}sft_debug_dataset.json")
        with open(debug_file_path, 'w') as debug_file:
            json.dump(data, debug_file, indent=4)

    return output_file


def process_conversation_for_sft(conversation, last_k_responses=None):
    """
    Processes a single conversation and formats it for SFT dataset extraction.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.

    Returns:
    - list: List of formatted entries for SFT fine-tuning.
    """
    context = []
    conversation_entries = []

    for message in conversation:
        # Skip messages with errors
        if message.get('is_error'):
            continue

        # Append user and assistant messages to the conversation context
        context.append(message)

        # Collect assistant responses
        if message.get('role') == "assistant":
            formatted_conversation = format_conversation_for_sft(copy.deepcopy(context))
            conversation_entries.append(formatted_conversation)

    # Limit to the last k assistant messages if specified
    if last_k_responses is not None:
        conversation_entries = conversation_entries[-last_k_responses:]

    return conversation_entries


def format_conversation_for_sft(context):
    """
    Formats a conversation into the desired format for SFT.

    Parameters:
    - context (list): List of messages in the conversation context.

    Returns:
    - dict: Formatted conversation with `messages` as a list of role-content pairs.
    """
    formatted_messages = []

    # Add system prompt (optional, you can customize this based on your needs)
    formatted_messages.append({"role": "system", "content": "You are helpful"})

    for message in context:
        role = message.get('role')
        content = message.get('content', '')

        if role in ["user", "assistant"]:
            formatted_messages.append({"role": role, "content": content})

    return {"messages": formatted_messages}
