import pandas as pd
import os
import re
from statistics import mean

def compute_dond_statistics(folder_path):
    game_stats = pd.DataFrame()  # Initialize an empty DataFrame for game statistics

    # Iterate over each file in the folder
    pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')
    
    for file_name in sorted(os.listdir(folder_path)):
        if pattern.match(file_name):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, index_col=0)

            # Transpose the DataFrame so that rows represent rounds and columns represent attributes
            df = df.transpose()

            # Convert string representations of dictionaries to actual dictionaries
            df['quantities'] = df['quantities'].apply(eval)
            df['player_0_values'] = df['player_0_values'].apply(eval)
            df['player_1_values'] = df['player_1_values'].apply(eval)
            df['player_0_finalization'] = df['player_0_finalization'].apply(eval)
            df['player_1_finalization'] = df['player_1_finalization'].apply(eval)
            df['agreement_reached'] = df['agreement_reached'].apply(eval)

            # Get number of rounds in game
            num_rounds = len(df)

            # Get total points of game
            player_0_total_points = df['player_0_points'].sum()
            player_1_total_points = df['player_1_points'].sum()

            # Calculate total_points_over_maximum for each row
            total_points_over_maximum_list = []
            for index, row in df.iterrows():
                maximum = 0
                p0_points = 0
                p1_points = 0
                if row['agreement_reached']:
                    for key in row['quantities'].keys():
                        maximum += max(row['player_0_values'][key], row['player_1_values'][key]) * row['quantities'][key]
                        p0_points += row['player_0_finalization'].get(key, 0) * row['player_0_values'][key]
                        p1_points += row['player_1_finalization'].get(key, 0) * row['player_1_values'][key]
                    total_points_over_maximum = (p0_points + p1_points) / maximum if maximum > 0 else 0
                else:
                    total_points_over_maximum = 0
                
                total_points_over_maximum_list.append(total_points_over_maximum)

            # Add total_points_over_maximum to the DataFrame
            df['total_points_over_maximum'] = total_points_over_maximum_list

            # Calculate and store statistics for the current game
            agreement_rate = df['agreement_reached'].sum() / num_rounds

            # Create a dictionary with the statistics for this game
            game_stat = {
                'player_0_total_points': int(player_0_total_points),
                'player_1_total_points': int(player_1_total_points),
                'agreement_rate': agreement_rate,
                'total_points_over_maximum': mean(total_points_over_maximum_list) if total_points_over_maximum_list else 0,
                'round_agreements': df['agreement_reached'].apply(lambda x: 1 if x else 0).tolist()  # This is a list
            }

            # Append these statistics as a new column in the game_stats DataFrame
            game_stats[file_name] = pd.Series(game_stat)

            # Save the updated DataFrame back to the CSV file (for individual games)
            df = df.transpose()
            df.to_csv(file_path, index=True)

    # Transpose the game_stats DataFrame so that each column is a game, and rows are statistics
    game_stats_file = os.path.join(folder_path, '2_game_stats.csv')
    game_stats.to_csv(game_stats_file, index=True)

    # Now, compute the mean of game_stats
    mean_game_stats = {}

    # Iterate over each row in the game_stats (each statistic type)
    for stat_name in game_stats.index:
        stat_values = game_stats.loc[stat_name]

        # If all values are numeric (scalar), calculate the mean directly
        if all(isinstance(val, (int, float)) for val in stat_values):
            mean_game_stats[stat_name] = mean(stat_values)

        # If there are lists (e.g., round_agreements), calculate the mean for each position
        elif isinstance(stat_values[0], list):
            # Find the maximum length of the lists to align them
            max_length = max(len(lst) for lst in stat_values if isinstance(lst, list))

            # Calculate the mean for each position in the lists
            mean_list = []
            for i in range(max_length):
                # Get the ith element from each list, or use None if the list is shorter
                ith_elements = [lst[i] for lst in stat_values if isinstance(lst, list) and len(lst) > i]

                # Calculate the mean for the ith elements, ignoring None
                mean_list.append(mean(ith_elements))

            mean_game_stats[stat_name] = mean_list

    # Convert the mean_game_stats to a DataFrame and save it
    mean_game_stats_df = pd.DataFrame(list(mean_game_stats.items()), columns=['stat_name', 'mean_value'])
    mean_game_stats_file = os.path.join(folder_path, '1_mean_game_stats.csv')
    mean_game_stats_df.to_csv(mean_game_stats_file, index=False)


