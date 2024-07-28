import json
import numpy as np
from prompts.instruction import get_instruction_prompt
from store import add_run_to_store
from type.behavior import Behavior
from utils import generate_initial_state
from logger import Logger
from DoND import DoND
from agents import DoNDagent
import hydra
import datetime
import os

class TwoPlayersNego:
    def __init__(self, iterations_per_run, games_per_iteration, game, player_0, player_1, logger):
        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.logger = logger

    def train_agents(self):
        pass

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            for _ in range(self.games_per_iteration):
                game_result = self.run_game()
                self.logger.log_game(game_result)
            self.train_agents()
            self.logger.new_iteration()

    def run_game(self):
        self.game.reset()
        ongoing = True
        message = None
        while ongoing:
            if self.game.current_turn() == "p0":
                message = self.player_0.play(message)
                ongoing = self.game.step(message)
            else:
                message = self.player_1.play(message)
                ongoing = self.game.step(message)
        return self.game.export()
    
class DoNDtrainer(TwoPlayersNegoTrainer):
    def train_agents(self):
        "Train the agents on the last iteration."
        metrics = self.log.metrics # extract dataframe with data for each game
        # train on all non zero 
        mean_score = self.log.iteration_stats['Mean Score'] # get the mean score of the current iteration
        # extract the file name of every game with score better than mean
        filtered_p0 = metrics[metrics['p0_score'] > mean_score] 
        filtered_p1 = metrics[metrics['p1_score'] > mean_score] 
        p0_filt_files = filtered_p0['p0_file'].tolist() 
        p1_filt_files = filtered_p1['p1_file'].tolist()
        p0_filt_jsons = [ json.load(file_path) for file_path in p0_filt_files ]
        p1_filt_jsons = [ json.load(file_path) for file_path in p1_filt_files ]
        self.player_0.train(p0_filt_jsons)
        self.player_1.train(p1_filt_jsons)

@hydra.main(config_path="../conf", config_name="config")
def RunDoND(cfg):

    # Make output directory
    out_dir = f"DATA/RUN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" 
    os.makedirs(out_dir, exist_ok=True)

    logger = Logger(out_dir)
    
    game = DoND()

    player_0 = DoNDagent(
        name="agent",
        device=cfg.device,
        model=cfg.p0.model,
        tokenizer=cfg.p0.tokenizer,
        chain_of_thought=cfg.p0.CoT,
        instructions=cfg.p0.force_conformity
    )

    player_1 = DoNDagent(
        name="agent",
        device=cfg.device,
        model=cfg.p1.model,
        tokenizer=cfg.p1.tokenizer,
        chain_of_thought=cfg.p1.CoT,
        instructions=cfg.p1.force_conformity
    )

    run_handler = TwoPlayersNego(
        iterations_per_run=cfg.run.nb_iterations,
        games_per_iteration=cfg.run.games_per_iterations,
        game=game,
        player_0=player_0,
        player_1=player_1,
        logger=logger
    )

    run_handler.run_iterations()

if __name__ == "__main__":
    RunDoND()
