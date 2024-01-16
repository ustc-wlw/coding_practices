
import sys
sys.path.append("C:\Workspace\develop\pytorch\torch_base_course\ch5_going_modular")
from typing import Dict, List
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ch5_going_modular.engine import train_step, test_step

def train(model: torch.nn.Module,
          train_dataloader:DataLoader,
          test_dataloader: DataLoader,
          optim: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device = torch.device("cpu"),
          writer: SummaryWriter = None
          ) -> Dict[str, list]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optim, device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        if writer:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss" : train_loss, "test_loss" : test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc" : train_acc, "test_acc" : test_acc},
                               global_step=epoch)
            writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))

    if writer:
            writer.close()

    return results