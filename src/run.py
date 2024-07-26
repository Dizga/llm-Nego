import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config")
def run(cfg: DictConfig):

    print(f"Training with config: {cfg}")
    model = hydra.utils.instantiate(cfg.model)
    print(f"Instantiated model: {model}")

if __name__ == "__main__":
    train()
