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
        self.game = game
        self.players = players
        self.models = models


        # Local
        self.iteration_nb = 0
        self.game_nb = 0
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
                        self.export_match(match['game'].export(), match['player_deque'])
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
        self.rounds_path = os.path.join(self.it_folder, 
                f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.csv")
        

    def export_match(self, game, players):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game List(dict): A list of dictionaries, each containing the data of a round.
        """

        # Create path
        game_name = f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}"

        # Export the player contexts
        for player in players:
            player_context_path = os.path.join(self.it_folder, f"{player.name}_{game_name}.json")
            os.makedirs(self.player_context_path, exist_ok=True)
            with open(player_context_path, 'w') as f:
                json.dump(player.get_context(), f, indent=4)

        # Export game metrics
        rounds_data = game.export()
        df = pd.DataFrame(rounds_data)
        df.set_index('round_id', inplace=True)
        df_transposed = df.transpose()
        df_transposed.to_csv(os.path.join(self.it_folder, f"{game_name}.json"))
            




