import os
import json
import logging
import logging.config
from datetime import datetime
import pandas as pd

class BcDondLogger:
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
        
        columns_metrics = [
            'p0_score',
            'p1_score',
            'quantities',
            'p0_values',
            'p1_values',
            'p0_proposal',
            'p1_proposal',
            'reach_agreement',
            'p0_file',
            'p1_file'
        ]
        self.metrics = pd.DataFrame(columns=columns_metrics)
        
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
        self.iteration = 0

    def new_iteration(self):
        """
        Starts a new iteration, resets metrics, and logs stats for the previous iteration.
        """
        self.iteration += 1
        self.game_nb = 0
        self.it_folder = os.path.join(self.run_dir, f"iteration_{self.iteration:02d}")
        os.makedirs(self.it_folder, exist_ok=True)
        # Reset metrics for the new iteration
        self.metrics = pd.DataFrame(columns=self.metrics.columns)
        self.metrics_file = os.path.join(self.it_folder, "metrics.csv")
        # Compute and log stats of the past iteration
        self.log_itr_stats()

    def get_itr_stats(self):
        """
        Computes statistics for the current iteration.

        Returns:
            dict: A dictionary containing statistics of the current iteration.
        """
        self.iteration_stats = {
            "Iteration": self.iteration,
            "Agreement Percentage": self.metrics['reach_agreement'].mean() * 100 if not self.metrics['reach_agreement'].empty else 0,
            "Score Variance P0": self.metrics['p0_score'].var() if not self.metrics['p0_score'].empty else 0,
            "Score Variance P1": self.metrics['p1_score'].var() if not self.metrics['p1_score'].empty else 0,
            "Mean Score P0": self.metrics['p0_score'].mean() if not self.metrics['p0_score'].empty else 0,
            "Mean Score P1": self.metrics['p1_score'].mean() if not self.metrics['p1_score'].empty else 0
        }
        return self.iteration_stats

    def log_itr_stats(self):
        """
        Logs statistics for the current iteration and saves them to a CSV file.
        """
        stats = self.get_itr_stats()
        self.statistics = pd.concat([self.statistics, pd.DataFrame([stats])], ignore_index=True)
        self.statistics.to_csv(self.statistics_file, index=False)

    def new_game(self):
        self.game_nb+=1

    def log_game(self, game: dict, p0_history, p1_history):
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

        game['p0_file'] = p0_game_name
        game['p1_file'] = p1_game_name

        # Log metrics
        self.metrics = pd.concat([self.metrics, pd.DataFrame([game])], ignore_index=True)
        self.metrics.to_csv(self.metrics_file, index=False)


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
