import json
import numpy as np
import hydra
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import sys
import os

from experiments.dond_run_train import dond_run_train
from experiments.arithmetic_test import arithmetic_test
#from experiments.last_completion import last_completion

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    globals()[cfg.experiment.method](cfg)

if __name__ == "__main__": main()

