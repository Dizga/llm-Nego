import json
from datetime import datetime
import os
import pandas as pd
import logging
import logging.config

# local imports
from environments.dond_game import DondGame
from utils.log_gpu_usage import log_gpu_usage

from collections import deque
import copy 
import time
import logging


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
        nb_matches = min(self.nb_parallel_games, self.games_per_iteration)
        for _ in range(nb_matches):
            match = {}
            match['player_list'] = [copy.deepcopy(player) for player in self.players]
            match['game'] = copy.deepcopy(self.game)
            match['game'].reset()
            match['game_state'] = match['game'].get_state()
            match['play_order'] = match['game'].get_play_order()
            match['player_deque'] = deque([match['player_list'][id] for id in match['play_order']])
            for i, player in enumerate(match['player_deque']): player.game_id = i
            self.matches.append(match)

        # Initiate prompt batches
        self.prompt_batches = {}
        self.response_batches = {}
        for model_name in self.models.keys():
            self.prompt_batches[model_name] = []
            self.response_batches[model_name] = []



    def run_iteration(self):

        iteration_name = f"Iteration {self.iteration_nb}"
        start_time = time.time()  # Start time for iteration

        self.new_iteration()
        logging.info(f"Iteration {self.iteration_nb} with {self.games_per_iteration} games started.")

        while self.game_nb < self.games_per_iteration:

            # Get prompt batch for each model
            for match in self.matches:

                # Add user message to context
                player = match['player_deque'][0]
                player.set_usr_message(match['game'].get_state())

                # Send player context to right model
                self.prompt_batches[player.model_name].append(copy.deepcopy(player.get_context()))

            # Process prompt batch of each model
            for model_name in self.models.keys():
                model = self.models[model_name]
                self.response_batches[model_name] = model.prompt(self.prompt_batches[model_name])
                assert len(self.response_batches[model_name]) == len(self.prompt_batches[model_name])
                self.prompt_batches[model_name] = []

            # Play moves for each player by using the model outputs
            for match in self.matches:

                player = match['player_deque'][0]
                response = self.response_batches[player.model_name].pop(0)
                send_to_game, is_finalization, processed_response = player.process_model_response(response, match['game_state'])

                # Player has made an official move (will be other player's turn next)
                if send_to_game:

                    match['player_deque'].rotate(1)

                    match['game_state'] = match['game'].step(processed_response, is_finalization)

                    if match['game_state']['round_ended']:
                        for player in match['player_list']: player.set_round_scores(match['game_state'])
                        match['play_order'] = match['game'].get_play_order()
                        match['player_deque'] = deque([match['player_list'][id] for id in match['play_order']])
                        for i, player in enumerate(match['player_deque']): player.game_id = i

                    if match['game_state']['game_ended']:
                        self.game_nb += 1
                        self.export_match(match['game'], match['player_deque'])
                        match['game'].reset()
                        for player in match['player_deque']: player.reset_game()

        # TODO: assert that the response batches now all have a size of 0
        
        end_time = time.time()
        iteration_duration = end_time - start_time
        logging.info(f"{iteration_name} completed in {iteration_duration:.2f} seconds.")
                    
                
                
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


        
        

    def export_match(self, game, players):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game List(dict): A list of dictionaries, each containing the data of a round.
        """

        logging.info(f"Game {self.game_nb} completed.")

        # Create path
        game_name = f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}"

        # Export the player contexts
        for player in players:
            player_context_path = os.path.join(self.it_folder, f"{player.player_name}_{game_name}.json")
            with open(player_context_path, 'w') as f:
                json.dump(player.get_context(), f, indent=4)

        # Export game metrics
        rounds_data = game.export()
        df = pd.DataFrame(rounds_data)
        df.set_index('round_id', inplace=True)
        df_transposed = df.transpose()
        df_transposed.to_csv(os.path.join(self.it_folder, f"{game_name}.csv"))
            




