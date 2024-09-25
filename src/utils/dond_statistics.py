import pandas as pd
import os
import re
from statistics import mean

def process_game_file(file_path):
    """
    Processes a single game CSV file and returns a dictionary of game statistics.
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path, index_col=0).transpose()

    # Convert string representations of dictionaries to actual dictionaries
    dict_columns = [
        'quantities', 'player_0_values', 'player_1_values',
        'player_0_finalization', 'player_1_finalization', 'agreement_reached'
    ]
    for col in dict_columns:
        df[col] = df[col].apply(eval)

    # Calculate total points for each player
    player_0_total_points = df['player_0_points'].sum()
    player_1_total_points = df['player_1_points'].sum()

    # Calculate total_points_over_maximum for each round
    total_points_over_maximum_list = []
    for _, row in df.iterrows():
        maximum = 0
        p0_points = 0
        p1_points = 0

        if row['agreement_reached']:
            for key in row['quantities']:
                quantity = row['quantities'][key]
                max_value = max(row['player_0_values'][key], row['player_1_values'][key])
                maximum += max_value * quantity
                p0_points += row['player_0_finalization'].get(key, 0) * row['player_0_values'][key]
                p1_points += row['player_1_finalization'].get(key, 0) * row['player_1_values'][key]
            total_points = p0_points + p1_points
            total_points_over_maximum = total_points / maximum if maximum > 0 else 0
        else:
            total_points_over_maximum = 0

        total_points_over_maximum_list.append(total_points_over_maximum)

    # Add the new column to the DataFrame
    df['total_points_over_maximum'] = total_points_over_maximum_list

    # Calculate agreement rate and round agreements
    num_rounds = len(df)
    agreement_rate = df['agreement_reached'].sum() / num_rounds
    round_agreements = df['agreement_reached'].astype(int).tolist()

    # Prepare game statistics
    game_stat = {
        'player_0_total_points': int(player_0_total_points),
        'player_1_total_points': int(player_1_total_points),
        'agreement_rate': agreement_rate,
        'total_points_over_maximum': mean(total_points_over_maximum_list) if total_points_over_maximum_list else 0,
        'round_agreements': round_agreements
    }

    # Save the updated DataFrame back to the CSV file
    df.transpose().to_csv(file_path, index=True)
    return game_stat

def compute_mean_game_stats(game_stats_list):
    """
    Computes the mean of the game statistics across all games.
    """
    mean_game_stats = {}
    scalar_stats = ['player_0_total_points', 'player_1_total_points', 'agreement_rate', 'total_points_over_maximum']

    # Convert the list of game stats dictionaries into a DataFrame
    game_stats_df = pd.DataFrame(game_stats_list)

    # Calculate mean for scalar statistics
    for stat in scalar_stats:
        mean_game_stats[stat] = game_stats_df[stat].mean()

    # Calculate mean for list statistics (e.g., round_agreements)
    round_agreements_lists = game_stats_df['round_agreements']
    max_length = max(len(lst) for lst in round_agreements_lists)
    mean_round_agreements = []

    for i in range(max_length):
        ith_elements = [lst[i] for lst in round_agreements_lists if len(lst) > i]
        mean_round_agreements.append(mean(ith_elements))

    mean_game_stats['round_agreements'] = mean_round_agreements
    return mean_game_stats

def compute_dond_statistics(folder_path):
    """
    Computes statistics for all game CSV files in the specified folder.
    Returns a dictionary of mean game statistics.
    """
    game_stats_list = []
    pattern = re.compile(r'^iter_\d{2}_game_\d{4}\.csv$')

    # Process each game file in the folder
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            file_path = os.path.join(folder_path, file_name)
            game_stat = process_game_file(file_path)
            game_stat['file_name'] = file_name  # Optional: include the file name
            game_stats_list.append(game_stat)

    # Save per-game statistics to a CSV file
    game_stats_df = pd.DataFrame(game_stats_list)
    game_stats_file = os.path.join(folder_path, '2_game_stats.csv')
    game_stats_df.to_csv(game_stats_file, index=False)

    # Compute and save mean game statistics
    mean_game_stats = compute_mean_game_stats(game_stats_list)
    mean_game_stats_df = pd.DataFrame([
        {'stat_name': key, 'mean_value': value} for key, value in mean_game_stats.items()
    ])

    # Convert lists to strings for CSV output
    mean_game_stats_df['mean_value'] = mean_game_stats_df['mean_value'].apply(
        lambda x: x if not isinstance(x, list) else str(x)
    )

    mean_game_stats_file = os.path.join(folder_path, '1_mean_game_stats.csv')
    mean_game_stats_df.to_csv(mean_game_stats_file, index=False)
    return mean_game_stats
