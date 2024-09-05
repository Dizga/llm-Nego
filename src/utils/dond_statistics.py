import pandas as pd
import os
import re
from statistics import *

def compute_dond_statistics(folder_path):

    game_stats = {
        'player_0_total_returns': [],
        'player_1_total_returns': [],
        'total_points_over_maximum': [], 
        'agreement_reached_percentage': [],
    } 

    # Iterate over each file in the folder
    pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')
    for file_name in sorted(os.listdir(folder_path)):
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

            # Calculate total_points_over_maximum for each row
            total_points_over_maximum_list = []
            for index, row in df.iterrows():
                maximum = 0
                p0_points = 0
                p1_points = 0
                # Calculate maximum and points
                if row['agreement_reached']:
                    for key in row['quantities'].keys():
                        maximum += max(row['player_0_values'][key], row['player_1_values'][key]) * row['quantities'][key]
                        p0_points += row['player_0_finalization'][key] * row['player_0_values'][key]
                        p1_points += row['player_1_finalization'][key] * row['player_1_values'][key]
                    total_points_over_maximum = (p0_points + p1_points) / maximum
                else:
                    total_points_over_maximum = 0 
                
                total_points_over_maximum_list.append(total_points_over_maximum)

            # Add total_points_over_maximum to the DataFrame
            df['total_points_over_maximum'] = total_points_over_maximum_list

            # Calculate and store statistics
            agreements_reached_percentage = df['agreement_reached'].sum() / num_rounds * 100

            game_stats['player_0_total_returns'].append(player_0_total_rewards)
            game_stats['player_1_total_returns'].append(player_1_total_rewards)
            game_stats['agreement_reached_percentage'].append(agreements_reached_percentage)
            game_stats['total_points_over_maximum'].extend(total_points_over_maximum_list)

            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)

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

