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
from utils.log_gpu_usage import log_gpu_usage

from collections import deque
import copy 


class DondIterationRunner:
    def __init__(self, 
                 out_dir,
                 nb_parallel_games,
                 games_per_iteration, 
                 game,
                 players,
                 models
                 ):

        self.out_dir = out_dir
        self.nb_parallel_games = nb_parallel_games
        self.games_per_iteration = games_per_iteration
        self.game
        self.players = players
        self.models = models


        # Local
        self.iteration_nb = 0
        self.game_nb = 0
        self.round_nb = 0
        self.run_dir = out_dir
        self.datenow = datetime.now().strftime('%Y_%m_%d_%H_%M')

        # Initiate parallel matches
        self.matches = []
        for _ in range(self.nb_parallel_games):
            match = {}
            match['player_list'] = copy.deepcopy(self.players)
            order = match['game'].get_play_order()
            match['game'] = copy.deepcopy(self.game)
            match['player_deque'] = deque([match['player_list'][id] for id in order])
            self.matches.append(match)



    def run_iteration(self):

        self.new_iteration()
        logging.info(f"Iteration {self.iteration_nb} with {self.game_nb} started.")

        while self.game_nb < self.games_per_iteration:

            # Get prompt batch for each model
            for match in self.matches:
                player = match['player_deque'][0]
                player.agent.append_to_input_batch(player.get_context())

            # Process prompt batch of each model
            for model in self.models:
                model.batched_responses = model.prompt(model.prompt_batch)

            # Play moves for each player by using the model outputs
            for match in self.matches:

                player = match['player_deque'][0]
                response = player.model.batch_response.pop(0)
                processed_move, send_to_game, is_finalization = player.process_model_response(response, match['game_state'])

                # Player has made an official move (will be other player's turn next)
                if send_to_game:

                    match['game_state'] = match['game'].step(processed_move, is_finalization)

                    if match['game_state']['game_ended']:
                        self.log_game(match['game'].export(), match['player_deque'])
                        logging.info(f"Game {self.game_nb} completed.")
                        for player in match['player_deque']: player.reset_game()
                        match['game'].reset()
                        self.reset_game()

                    elif match['game_state']['round_ended']:
                        play_order = match['game'].get_play_order()
                        match['player_deque'] = deque([match['player_list'][id] for id in play_order])
                        for player in match['player_deque']: player.reset_round()

                    match['player_deque'].rotate(1)
                    
                
                
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

    def reset_game(self):
        self.game_nb += 1
        self.round_nb = 0
        self.rounds_log = pd.DataFrame([])
        self.rounds_path = os.path.join(self.it_folder, 
                f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.csv")
        

    def log_game(self, game, player_0_history, player_1_history):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game List(dict): A list of dictionaries, each containing the data of a round.
        """
        # TODO: refactor
        
        # Export the conversations
        player_0_game_name = f"player_0_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"
        player_1_game_name = f"player_1_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.it_folder, player_0_game_name), 'w') as f:
            json.dump(player_0_history, f, indent=4)
        with open(os.path.join(self.it_folder, player_1_game_name), 'w') as f:
            json.dump(player_1_history, f, indent=4)

        # Log every round
        for round in game: self.log_round(round)


    def log_round(self, round: dict):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        # TODO: refactor

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
        # TODO: refactor

        file_path = os.path.join(self.run_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
