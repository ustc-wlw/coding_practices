import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from pathlib import Path
import os

# cpu_numbers = os.cpu_count()
cpu_numbers = 1

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: transforms.Compose,
                       test_transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int=cpu_numbers):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform,
                                      target_transform=None)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_data = datasets.ImageFolder(root=test_dir,
                                      transform=test_transform,
                                      target_transform=None)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'dataset train samples: {len(train_data)}, test sample: {len(test_data)}')

    return train_dataloader, test_dataloader, train_data.classes

if __name__=='__main__':
    train_dir = "../ch4_dataset/data/pizza_steak_sushi/train"
    test_dir = "../ch4_dataset/data/pizza_steak_sushi/test"

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()]
    )

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()]
    )

    batch_size = 32

    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        train_transform=train_transform,
                                                                        test_transform=test_transform,
                                                                        batch_size=batch_size)
    print(f'class names: {class_names}')

    imgs, labels = next(iter(test_dataloader))
    print(f'img shape: {imgs.shape}, batch label len: {len(labels)}, labels: {labels}')