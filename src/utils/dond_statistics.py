import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from statistics import *

def compute_dond_statistics(folder_path):
    # Initialize lists to store statistics and dictionaries for round scores
    total_returns = {'player_0':[], 'player_1':[]}
    round_scores = defaultdict(lambda: defaultdict(list))
    agreements_reached = []
    num_rounds = 0

    # Define the regex pattern for matching filenames
    pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Convert string representations of dictionaries to actual dictionaries
            df['quantities'] = df['quantities'].apply(eval)
            df['player_0_values'] = df['player_0_values'].apply(eval)
            df['player_1_values'] = df['player_1_values'].apply(eval)
            df['player_0_proposal'] = df['player_0_proposal'].apply(eval)
            df['player_1_proposal'] = df['player_1_proposal'].apply(eval)

            # Compute statistics for this file
            total_returns['player_0'].append(df['player_0_reward'].sum())
            total_returns['player_1'].append(df['player_1_reward'].sum())

            
            for _, row in df.iterrows():
                round_id = row['round_id']
                round_scores['player_0'][round_id].append(row['player_0_reward'])
                round_scores['player_1'][round_id].append(row['player_1_reward'])

            agreements_reached.extend(df['agreement_reached'])
            num_rounds += len(df)

    # Compute round scores
    round_score_list = {
        player: {round_id: np.mean(scores) for round_id, scores in scores_dict.items()}
        for player, scores_dict in round_scores.items()
    }

    # Compute desired statistics
    stats = {
        'Mean Total Return Player 0': mean(total_returns['player_0']),
        'Mean Total Return Player 1': mean(total_returns['player_1']),
        # 'Mean Round Return Variance': np.mean([np.var(list(scores.values())) for scores in round_score_list.values()]),
        #'Variance for Total Return': np.var(total_returns),
        '% Agreements Reached Per Round': mean(agreements_reached) * 100
    }

    # Create a DataFrame from the dictionary and save as CSV
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    output_file = os.path.join(folder_path, 'global_iteration_statistics.csv')
    stats_df.to_csv(output_file, index=False)


