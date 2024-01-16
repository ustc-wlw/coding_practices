import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_func: nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device=torch.device("cpu")
):
    train_loss, train_acc = 0, 0
    model.train()
    # model.to(device)
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_func(logits, y)
        train_loss += loss.item()

        optim.zero_grad()

        loss.backward()

        optim.step()

        pred_label = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = torch.eq(pred_label, y).sum().item() / len(y)
        train_acc += acc
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: DataLoader,
        loss_func: nn.Module,
        device: torch.device=torch.device("cpu")
):
    test_loss, test_acc = 0, 0
    model.eval()
    # model.to(device)
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_func(logits, y)
            test_loss += loss.item()
            pred_label = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            acc = torch.eq(pred_label, y).sum().item() / len(y)
            test_acc += acc

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optim: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device=torch.device("cpu")):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_func=loss_fn, optim=optim, device=device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_func=loss_fn, device=device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f'Epoch {epoch + 1} Training loss: {train_loss:.4f} | Training acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')

    return results
