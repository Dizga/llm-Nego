import json
import hydra
import os
import logging
import logging.config
import pandas as pd

from environments.ultimatum import Ultimatum
from players.ultimatum_players import UltimatumPlayer
from players.get_players import setup_players


class UltimatumIterator():
    def __init__(self,
                 out_dir,
                 games_per_iteration, 
                 game: Ultimatum,
                 players: list[UltimatumPlayer],
                 ):

        self.games_per_iteration = games_per_iteration
        self.game = game
        self.players = players
        self.run_dir = out_dir

        self.iteration_nb = 0
        self.game_nb = 0
        self.round_nb = 0

    def run_iteration(self):
        self.new_iteration()
        for _ in range(self.games_per_iteration):
            self.run_game()

    def run_game(self):
        logging.info("Game started.")
        game_state = self.new_game()
        player_id = 0

        while not game_state['game_ended']:
            content = self.players[player_id].play_move(game_state)
            game_state = self.game.step(content)
            player_id = (player_id + 1) % 2

        self.log_game(*self.game.export(), 
                             self.players[0].get_history(), 
                             self.players[1].get_history())

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
        for player in self.players:
            player.new_game()
        return self.game.reset()

    def log_game(self, game, player_0_history, player_1_history):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game List(dict): A list of dictionaries, each containing the data of a round.
        """
        
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
        # Log round metrics
        self.rounds_log = pd.concat([self.rounds_log, pd.DataFrame([round])], ignore_index=True)
        self.rounds_log.to_csv(self.rounds_path, index=False)

    def _start_new_round(self):
        for player in self.players:
            player.new_round()


def ultimatum(cfg): 
    
    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    dond_game = Ultimatum(**cfg.game)

    players = setup_players(cfg, player_type=UltimatumPlayer)

    iteration_runner = UltimatumIterator(
        output_directory,
        cfg.playing.games_per_iteration, 
        game=dond_game,
        players=players,
    )

    for _ in range(cfg.playing.nb_iterations):
        
        # Play games
        iteration_runner.run_iteration()



