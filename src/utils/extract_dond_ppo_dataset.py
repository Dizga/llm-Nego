import json
import os
import pandas as pd
import regex as re
import copy

def extract_ppo_dataset(folder_path: str, 
                           player_name, 
                           export_for_debugging=True):
    """
    Extracts data for HF PPO training from game logs.

    Args:
        folder_path (str): Path to the folder containing game logs.
        player_0 (bool): If True, extracts data for player 0, otherwise for player 1.
        full_context (bool): If True, includes full context of the conversation.
        export_for_debugging (bool): If True, exports the queries, responses, and scores to a JSON file for debugging.

    Returns:
        tuple: Lists of queries, responses, and scores.
    """

    player_prefix = player_name + "_" 
    queries, responses, scores = [], [], []

    # For each game player
    for file_name in os.listdir(folder_path):
        
        pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')

        context = []

        if pattern.match(file_name):

            # Get list of rewards
            game = pd.read_csv(os.path.join(folder_path, file_name))
            game = game.T
            rewards = game[player_prefix + "reward"].tolist()

            # Import conversation
            conversation_path = os.path.join(folder_path, player_prefix+file_name.replace(".csv", ".json"))
            with open(conversation_path, 'r') as file: conversation = json.load(file)

            # Extract queries, responses, and scores
            context = []
            count = -1

            # extract queries, responses and scores
            for message in conversation:

                # Don't add mistakes to training data
                if message['is_error']: continue

                context.append(message)

                # An action has been made
                if message['role'] == "assistant":

                    queries.append(context[:-1])

                    responses.append([message])

                    scores.append(rewards[count])

                if message.get('is_new_round'):
                    count += 1

            

    # Export to facilitate debugging
    if export_for_debugging:
        debug_data = [{"query": q, "response": r, "score": s} 
                      for q, r, s in zip(queries, responses, scores)]
        
        debug_file_path = os.path.join(folder_path, "extracted_training_dataset.json")
        with open(debug_file_path, 'w') as debug_file:
            json.dump(debug_data, debug_file, indent=4)

    return queries, responses, scores
