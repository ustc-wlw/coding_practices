import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

def create_writer(experienment_name: str,
                  model_name: str,
                  extra: str=None) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y-%m-%d")
    print(f'current timestamp is {timestamp}')

    if extra:
        log_dir = os.path.join("runs", experienment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", experienment_name, model_name)

    print(f"[info] Create SummaryWriter, saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    example_writer = create_writer(experienment_name="data_10_percent",
                                   model_name="effnetb0",
                                   extra="5_epochs")
    print(f'writer is {example_writer}')

