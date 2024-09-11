import json
import os
import pandas as pd
import regex as re
import copy
from statistics import mean

# TODO: parameterize this function so that it can be adjusted with the config


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

            # TODO: remove if want to train on games with no agreements!
            if conversation[-1]['self_score'] == 0:
                continue

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


    # TODO: determine customize!
    if len(scores) > 0:
        mean_score = mean(scores)
        scores = [s - mean_score for s in scores]

            

    # Export to facilitate debugging
    if export_for_debugging:
        debug_data = [{"query": q, "response": r, "score": s} 
                      for q, r, s in zip(queries, responses, scores)]
        
        debug_file_path = os.path.join(folder_path, f"{player_prefix}extracted_training_dataset.json")
        with open(debug_file_path, 'w') as debug_file:
            json.dump(debug_data, debug_file, indent=4)

    return queries, responses, scores
