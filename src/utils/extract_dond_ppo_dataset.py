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

    """

    player_prefix = player_name + "_" 
    queries, responses, scores = [], [], []

    for file_name in os.listdir(folder_path):
        
        pattern = re.compile(rf'^{re.escape(player_prefix)}iter_\d{{2}}_game_\d{{4}}\.json$')

        context = []

        if pattern.match(file_name):

            # Import conversation
            conversation_path = os.path.join(folder_path, file_name)
            with open(conversation_path, 'r') as file: conversation = json.load(file)

            # Extract queries, responses, and scores
            context = []

            # extract queries, responses and scores
            for message in conversation:

                # Don't add mistakes to training data
                if message['is_error']: continue

                context.append(message)

                # An action has been made
                if message['role'] == "assistant":

                    queries.append(context[:-1])

                    responses.append([message])

                    scores.append(message['self_score'])

                # if message.get('is_new_round'):
                #    count += 1

            

    # Export to facilitate debugging
    if export_for_debugging:
        debug_data = [{"query": q, "response": r, "score": s} 
                      for q, r, s in zip(queries, responses, scores)]
        
        debug_file_path = os.path.join(folder_path, "extracted_training_dataset.json")
        with open(debug_file_path, 'w') as debug_file:
            json.dump(debug_data, debug_file, indent=4)

    return queries, responses, scores
