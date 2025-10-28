from dataclasses import dataclass

@dataclass
class TrainArgs:
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-4
    device: int = 0
    world_size: int = 1

class Trainer:
    def __init__(self, config: TrainArgs) -> None:
        # initialize model
        # initialize dataloader
        # load and save checkpoints
        pass

    def train(self):
        # training loop
        pass

    def load_snapshot(self):
        # load a checkpoint
        pass

    def save_snapshot(self):
        # save a checkpoint
        pass

"""
design:
trainer is the work horse of model training
it will initialize the model, dataloader, run training, save and load checkpoints
"""