import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pandas as pd
import logging
import logging.config

# local imports
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent


class DondIterationRunner:
    def __init__(self, 
                 out_dir,
                 games_per_iteration, 
                 game: DondGame, 
                 player_0: DondPlayer, 
                 player_1: DondPlayer, 
                 ):

        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1

        self.run_dir = out_dir
        self.datenow = datetime.now().strftime('%Y_%m_%d_%H_%M')

        self.iteration_nb = 0
        self.game_nb = 0
        self.round_nb = 0

    def run_iteration(self):
        self.new_iteration()
        for _ in range(self.games_per_iteration):
            self.run_game()

    def run_game(self):
        logging.info(f"Game {self.game_nb} of iteration {self.iteration_nb} started.")
        self.new_game()
        players = [self.player_0, self.player_1]
        self.player_0.new_game()
        self.player_1.new_game()
        game_state = self.game.reset()
        player_id = 0
        while not game_state['game_ended']:
            if game_state['new_round']:
                self.player_0.new_round()
                self.player_1.new_round()
            is_proposal, content = players[player_id].play_move(game_state)
            game_state = self.game.step(content, is_proposal=is_proposal)
            player_id = (player_id + 1) % 2
            
        # while True:
        self.log_game(*self.game.export(), 
                             self.player_0.get_history(), 
                             self.player_1.get_history())
        logging.info("Game completed.")

    def new_iteration(self)-> str:
        """
        Starts a new iteration, resets metrics, and logs stats for the previous iteration.
        Returns:
            Path of folder where data is being logged.
        """
        self.iteration_nb += 1
        self.game_nb = 0
        self.it_folder = os.path.join(self.run_dir, f"iteration_{self.iteration_nb:02d}")
        os.makedirs(self.it_folder, exist_ok=True)
        # Reset metrics for the new iteration
        self.game_log = pd.DataFrame()
        self.game_log_file = os.path.join(self.it_folder, "games.csv")
        return self.it_folder

    def new_game(self):
        self.game_nb += 1
        self.round_nb = 0
        self.rounds_log = pd.DataFrame([])
        self.rounds_path = os.path.join(self.it_folder, 
                f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.csv")
        

    def log_game(self, summary, rounds, player_0_history, player_1_history):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        
        player_0_game_name = f"player_0_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"
        player_1_game_name = f"player_1_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"

        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.it_folder, player_0_game_name), 'w') as f:
            json.dump(player_0_history, f, indent=4)

        with open(os.path.join(self.it_folder, player_1_game_name), 'w') as f:
            json.dump(player_1_history, f, indent=4)

        summary['player_0_path'] = player_0_game_name
        summary['player_1_path'] = player_1_game_name
        summary['rounds_path'] = self.rounds_path

        # Log global game metrics
        self.game_log = pd.concat([self.game_log, pd.DataFrame([summary])], ignore_index=True)
        self.game_log.to_csv(self.game_log_file, index=False)

        # Log every round
        for round in rounds: self.log_round(round)


    def log_round(self, round: dict):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        # Log round metrics
        self.rounds_log = pd.concat([self.rounds_log, pd.DataFrame([round])], ignore_index=True)
        self.rounds_log.to_csv(self.rounds_path, index=False)

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