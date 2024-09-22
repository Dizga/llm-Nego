import json
import os
import copy
from statistics import mean
import regex as re

def extract_ppo_dataset(folder_path: str, 
                        player_name="bob", 
                        export_for_debugging=True, 
                        use_pattern_matching=True,
                        substract_mean=False,
                        last_k_responses=None):
    """
    Extracts data for HF PPO training from game logs.

    Parameters:
    - folder_path (str): Path to the folder containing conversation JSON files.
    - player_name (str): Name of the player or agent.
    - export_for_debugging (bool): If True, exports the extracted data for debugging.
    - use_pattern_matching (bool): If True, processes files matching the specific pattern.
                                   If False, processes all JSON files in the folder.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.
    """
    player_prefix = player_name + "_"
    queries, responses, scores = [], [], []

    # Define the pattern if pattern matching is enabled
    if use_pattern_matching:
        pattern = re.compile(rf'^{re.escape(player_prefix)}iter_\d{{2}}_game_\d{{4}}\.json$')

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

        # Process conversation
        conv_queries, conv_responses, conv_scores = process_conversation(
            conversation, last_k_responses=last_k_responses
        )

        queries.extend(conv_queries)
        responses.extend(conv_responses)
        scores.extend(conv_scores)

    # Adjust scores by subtracting the mean
    if substract_mean and scores:
        mean_score = mean(scores)
        scores = [s - mean_score for s in scores]

    # Export to facilitate debugging
    if export_for_debugging:
        debug_data = [{"query": q, "response": r, "score": s} 
                      for q, r, s in zip(queries, responses, scores)]
        
        debug_file_name = f"{player_prefix}extracted_training_dataset.json"
        debug_file_path = os.path.join(folder_path, debug_file_name)
        with open(debug_file_path, 'w') as debug_file:
            json.dump(debug_data, debug_file, indent=4)

    return queries, responses, scores

def process_conversation(conversation, last_k_responses=None):
    """
    Processes a single conversation and extracts queries, responses, and scores.

    Parameters:
    - conversation (list): List of message dictionaries representing a conversation.
    - last_k_responses (int or None): If set, only the last k assistant messages will be trained on.
                                      If None, all messages are considered.

    Returns:
    - conversation_queries (list): List of queries (contexts) extracted from the conversation.
    - conversation_responses (list): List of assistant responses.
    - conversation_scores (list): List of scores associated with the responses.
    """
    context = []
    conversation_queries = []
    conversation_responses = []
    conversation_scores = []

    # Optionally skip conversations with no agreements
    # if conversation[-1].get('self_score', 0) == 0:
    #     return [], [], []

    for message in conversation:
        # Skip messages with errors
        if message.get('is_error'):
            continue

        context.append(message)

        # Collect assistant responses
        if message.get('role') == "assistant":
            # Use deepcopy to avoid modifying the context elsewhere
            conversation_queries.append(copy.deepcopy(context[:-1]))
            # TODO: perhaps put back to normal
            conversation_responses.append([message])
            conversation_scores.append(message.get('self_score', 0))

    # Limit to the last k assistant messages if specified
    if last_k_responses is not None:
        conversation_queries = conversation_queries[-last_k_responses:]
        conversation_responses = conversation_responses[-last_k_responses:]
        conversation_scores = conversation_scores[-last_k_responses:]

    return conversation_queries, conversation_responses, conversation_scores
