import json
import numpy as np
import hydra
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import sys
import os

from experiments.dond_ppo_run_train_cycle import dond_ppo_run_train_cycle
#from experiments.ultimatum_run import ultimatum
from experiments.training_tester import training_tester

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    training_tester(cfg)
if __name__ == "__main__": main()

