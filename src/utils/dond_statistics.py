import pandas as pd
import os
import re
from statistics import *

def compute_dond_statistics(folder_path):

    game_stats = {
        'player_0_total_returns': [],
        'player_1_total_returns': [],
        'total_reward_over_coop_optimum': [], 
        'agreement_reached_percentage': [],
    } 

    # Iterate over each file in the folder
    pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Get number of rounds in game
            num_rounds = len(df)

            # Convert string representations of dictionaries to actual dictionaries
            df['quantities'] = df['quantities'].apply(eval)
            df['player_0_values'] = df['player_0_values'].apply(eval)
            df['player_1_values'] = df['player_1_values'].apply(eval)
            df['player_0_finalization'] = df['player_0_finalization'].apply(eval)
            df['player_1_finalization'] = df['player_1_finalization'].apply(eval)

            # Get total rewards of game
            player_0_total_rewards = df['player_0_reward'].sum()
            player_1_total_rewards = df['player_1_reward'].sum()

            # Perfect cooperation
            coop_optimum = 0
            for index, row in df.iterrows():
                p0vs = row['player_0_values']
                p1vs = row['player_1_values']
                for key in p0vs:
                    coop_optimum += max(p0vs[key], p1vs[key]) * row['quantities'][key]
            total_reward_over_coop_optimum = player_0_total_rewards / coop_optimum

            agreements_reached_percentage = df['agreement_reached'].sum()/num_rounds * 100

            game_stats['player_0_total_returns'].append(player_0_total_rewards)
            game_stats['player_1_total_returns'].append(player_1_total_rewards)
            game_stats['total_reward_over_coop_optimum'].append(total_reward_over_coop_optimum)
            game_stats['agreement_reached_percentage'].append(agreements_reached_percentage)


    # Compute desired statistics
    global_game_stats = {}
    for key in game_stats.keys():
        global_game_stats["mean_" + key] = mean(game_stats[key])

    # Create DataFrames from the dictionaries
    game_stats_df = pd.DataFrame(list(game_stats.items()), columns=['Statistic', 'Value'])
    global_stats_df = pd.DataFrame(list(global_game_stats.items()), columns=['Statistic', 'Value'])

    # Save the DataFrames as separate CSV files
    game_stats_file = os.path.join(folder_path, 'game_stats.csv')
    global_stats_file = os.path.join(folder_path, 'global_stats.csv')
    
    game_stats_df.to_csv(game_stats_file, index=False)
    global_stats_df.to_csv(global_stats_file, index=False)

    print(f"Statistics exported to {game_stats_file} and {global_stats_file}")

