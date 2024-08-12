import os
import json
import logging
import logging.config
from datetime import datetime
import pandas as pd
import torch


class DondLogger:
    """
    Logger class for logging game data, metrics, and statistics.
    """
    def __init__(self, out_dir):
        """
        Initializes the Logger.

        Args:
            out_dir (str): The output directory where logs will be saved.
        """
        self.run_dir = out_dir
        self.datenow = datetime.now().strftime('%Y_%m_%d_%H_%M')

        columns_statistics = [
            "Iteration",
            "Agreement Percentage",
            "Score Variance P0",
            "Score Variance P1",
            "Mean Score P0",
            "Mean Score P1"
        ]
        self.statistics = pd.DataFrame(columns=columns_statistics)
        self.statistics_file = os.path.join(self.run_dir, "stats.csv")
        self.iteration_nb = 0
        self.game_nb = 0
        self.round_nb = 0

    def new_iteration(self)-> str:
        """
        Starts a new iteration, resets metrics, and logs stats for the previous iteration.
        Returns:
            Path of folder where data is being logged.
        """
        self.iteration += 1
        self.game_nb = 0
        self.it_folder = os.path.join(self.run_dir, f"iteration_{self.iteration:02d}")
        os.makedirs(self.it_folder, exist_ok=True)
        # Reset metrics for the new iteration
        self.game_log = pd.DataFrame()
        self.game_log_file = os.path.join(self.it_folder, "metrics.csv")
        return self.it_folder

    def set_itr_stats(self):
        """
        Computes statistics for the current iteration.

        Returns:
            dict: A dictionary containing statistics of the current iteration.
        """
        self.iteration_stats = {
            "Iteration": self.iteration,
            "Agreement Percentage": self.game_log['agreement_reached'].mean() * 100 if not self.game_log['agreement_reached'].empty else 0,
            "Score Variance P0": self.game_log['p0_score'].var() if not self.game_log['p0_score'].empty else 0,
            "Score Variance P1": self.game_log['p1_score'].var() if not self.game_log['p1_score'].empty else 0,
            "Mean Score P0": self.game_log['p0_score'].mean() if not self.game_log['p0_score'].empty else 0,
            "Mean Score P1": self.game_log['p1_score'].mean() if not self.game_log['p1_score'].empty else 0
        }

    def log_itr_stats(self):
        """
        Logs statistics for the current iteration and saves them to a CSV file.
        """
        self.set_itr_stats()
        iteration = self.iteration_stats['Iteration']

        if iteration in self.statistics['Iteration'].values:
            self.statistics.loc[self.statistics['Iteration'] == iteration, :] = pd.DataFrame([self.iteration_stats])
        else:
            self.statistics = pd.concat([self.statistics, pd.DataFrame([self.iteration_stats])], ignore_index=True)
        self.statistics.to_csv(self.statistics_file, index=False)


    def new_game(self):
        self.game_nb += 1
        self.round_nb = 0
        self.rounds_log = pd.DataFrame([])
        self.rounds_path = os.path.join(self.it_folder, 
                f"iter_{self.iteration:02d}_game_{self.game_nb:04d}.json")
        

    def log_game(self, summary, rounds, p0_history, p1_history):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        
        p0_game_name = f"p0_iter_{self.iteration:02d}_game_{self.game_nb:04d}.json"
        p1_game_name = f"p1_iter_{self.iteration:02d}_game_{self.game_nb:04d}.json"

        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.it_folder, p0_game_name), 'w') as f:
            json.dump(p0_history, f, indent=4)

        with open(os.path.join(self.it_folder, p1_game_name), 'w') as f:
            json.dump(p1_history, f, indent=4)

        summary['p0_path'] = p0_game_name
        summary['p1_path'] = p1_game_name
        summary['rounds_path'] = self.rounds_path

        # Log global game metrics
        self.game_log = pd.concat([self.game_log, pd.DataFrame([game])], ignore_index=True)
        self.game_log.to_csv(self.game_log_file, index=False)

        # Log every round
        for round in rounds:
            self.log_round(self, round)

        # Adjust iteration statistics (even if not finished)
        self.log_itr_stats()

    def log_round(self, round: dict):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        # Log round metrics
        self.rounds_log = pd.concat([self.rounds_log, pd.DataFrame([round])], ignore_index=True)
        self.rounds_log.to_csv(self.rounds_path, index=False)

        # Adjust iteration statistics (even if not finished)
        self.log_itr_stats()

    def extract_hf_ppo_dataset(self, folder_path: str, p0=True, full_context=True):
        """
        Args:
            file (str): Location of the csv / dataframe for the iteration
        """

        if p0: 
            gm_messages_path_df_column = "p0_messages_path"
            mg_rewards_df_column = "p0_return"
        else: 
            gm_messages_path_df_column = "p1_messages_path"
            mg_rewards_df_column = "p1_return"

        # get jsons list
        queries = []
        responses = []
        scores = []

        # get all the games 
        games_info_df = pd.read_csv(os.path.join(folder_path, 'games.csv')) 
        games_info = games_info_df.to_dict(orient='records')

        # TODO: only analyse the games from the right player

        for game_info in games_info:

            # get game returns
            game_path = game_info[gm_messages_path_df_column]
            rounds_metrics_df = pd.read_csv(game_info['rounds_metrics_path'])  # get rounds dataframe
            mg_rewards = rounds_metrics_df[mg_rewards_df_column].tolist()

            # get game conversation
            with open(os.path.join(game_path, 'json'), 'r') as file:
                game = json.load(file)

            context = []
            count = 0

            # extract queries, responses and scores
            for message in game:
                if message['role'] == "assistant":
                    queries.append(context)
                    responses.append(message)
                    scores.append(mg_rewards[count])
                elif message['is_new_round']:
                    count += 1
                context.append(message)

        return queries, responses, scores



    def setup_logging(self, config_file):
        """
        Sets up logging configuration from a config file.

        Args:
            config_file (str): Path to the logging configuration file.
        """
        logging.config.fileConfig(config_file, defaults={'date': self.datenow})

    def log_info(self, message: str):
        """
        Logs an info message.

        Args:
            message (str): The message to log.
        """
        logging.info(message)

    def log_debug(self, message: str):
        """
        Logs a debug message.

        Args:
            message (str): The message to log.
        """
        logging.debug(message)

    def log_warning(self, message: str):
        """
        Logs a warning message.

        Args:
            message (str): The message to log.
        """
        logging.warning(message)

    def log_error(self, message: str):
        """
        Logs an error message.

        Args:
            message (str): The message to log.
        """
        logging.error(message)

    def save_player_messages(self, player_name: str, messages: list):
        """
        Saves player messages to a JSON file.

        Args:
            player_name (str): The name of the player.
            messages (list): A list of messages from the player.
        """
        file_path = os.path.join(self.run_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
