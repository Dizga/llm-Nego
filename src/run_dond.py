import os
import json
import hydra
from omegaconf import OmegaConf
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_instructor import DondInstructor
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent

class DondTrainer:
    def __init__(self, game: DondGame, instructors, logger: DondLogger, train_type: str):
        self.game = game
        self.instructors = instructors
        self.logger = logger
        self.train_type = train_type

    def play_games(self, iterations, games_per_iteration):
        for iteration in range(iterations):
            folder_path = self.logger.new_iteration()
            for _ in range(games_per_iteration):
                self._play_single_game()
            if self.train_type == "ppo":
                self.train_ppo(folder_path)

    def _play_single_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        game_state = self.game.reset()
        player_id = 0

        while not game_state['game_ended']:
            if game_state['new_round']:
                self._start_new_round()
            is_proposal, content = self.instructors[player_id].play_move(game_state)
            game_state = self.game.step(content, is_proposal=is_proposal)
            player_id = (player_id + 1) % 2

        self.logger.log_game(*self.game.export(), 
                             self.instructors[0].get_history(), 
                             self.instructors[1].get_history())
        self.logger.log_info("Game completed.")

    def _start_new_round(self):
        for instructor in self.instructors:
            instructor.new_round()

    def train_ppo(self, folder_path):
        self.logger.log_info("PPO training started.")
        for i, instructor in enumerate(self.instructors):
            instructor.dond_player.init_ppo_trainer()
            queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, p0=(i == 0))
            instructor.dond_player.train_ppo_json(queries, responses, scores)
        self.logger.log_info("PPO training ended.")

    def train_bc(self, folder_path):
        metrics = self.logger.metrics
        for i, instructor in enumerate(self.instructors):
            mean_score = self.logger.iteration_stats[f'Mean Score P{i}']
            filtered_metrics = metrics[metrics[f'p{i}_score'] >= mean_score]
            filtered_files = [os.path.join(folder_path, f) for f in filtered_metrics[f'p{i}_file'].tolist()]
            filtered_jsons = [json.load(open(file_path, 'r')) for file_path in filtered_files]
            instructor.dond_player.train(filtered_jsons)

def setup_instructors(cfg, game):
    agents = []
    for player_cfg in [cfg.players.p0, cfg.players.p1]:
        if player_cfg.type == "hf":
            agent = HfAgent(**player_cfg.agent_args)
        elif player_cfg.type == "dummy_hf":
            agent = DummyHfAgent(**player_cfg.agent_args)
        elif player_cfg.type == "oai":
            agent = OaiAgent(**player_cfg.agent_args)
        agents.append(agent)
    
    if cfg.players.shared_model:
        agents[1].model = agents[0].model

    instructors = [
        DondInstructor(**cfg.players.p0.instructor_args, dond_game=game, dond_player=agents[0], player_type="p0"),
        DondInstructor(**cfg.players.p1.instructor_args, dond_game=game, dond_player=agents[1], player_type="p1")
    ]
    return instructors

def run_dond(cfg):
    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = DondLogger(output_directory)
    game = DondGame(**cfg.game)
    instructors = setup_instructors(cfg, game)
    trainer = DondTrainer(game, instructors, logger, cfg.training.train_type)

    if cfg.training.skip_game_play:
        folder_path = cfg.training.load_folder
        if cfg.training.train_type == "ppo":
            trainer.train_ppo(folder_path)
        else:
            trainer.train_bc(folder_path)
    else:
        trainer.play_games(cfg.training.iterations_per_run, cfg.training.games_per_iteration)
        if cfg.training.train_type == "bc":
            trainer.train_bc(logger.iteration_folder)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    if os.path.exists('conf/local.yaml'):
        local_cfg = OmegaConf.load('conf/local.yaml')
        cfg = OmegaConf.merge(cfg, local_cfg)
    run_dond(cfg)

if __name__ == "__main__":
    main()
