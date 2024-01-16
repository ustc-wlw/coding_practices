import torch
from torchvision import transforms

import sys
sys.path.append("C:\Workspace\develop\pytorch\torch_base_course")
from ch5_going_modular import dataset_setup
from models import create_effnetb0, create_effnetb2
from engine import train
from utils import create_writer

from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader_10_percent, train_dataloader_20_percent, test_dataloader = None, None, None
class_names = None
def set_seeds(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_dataset():
    global train_dataloader_10_percent, train_dataloader_20_percent, test_dataloader
    global class_names
    BATCH_SIZE = 32

    data_20_percent_path = Path("./dataset")
    data_10_percent_path = Path("../ch4_dataset/data/pizza_steak_sushi")

    train_dir_10_percent = data_10_percent_path / "train"
    test_dir = data_10_percent_path / "test"
    train_dir_20_percent = data_20_percent_path / "train"

    print(f"Training directory 10%: {train_dir_10_percent}")
    print(f"Training directory 20%: {train_dir_20_percent}")
    print(f"Testing directory: {test_dir}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    manul_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataloader_10_percent, test_dataloader, class_names = dataset_setup.create_dataloaders(train_dir=train_dir_10_percent, test_dir=test_dir,
                                                                   train_transform=manul_transform,
                                                                   test_transform=manul_transform,
                                                                   batch_size=BATCH_SIZE)

    train_dataloader_20_percent, _, _ = dataset_setup.create_dataloaders(train_dir=train_dir_20_percent, test_dir=test_dir,
                                                                   train_transform=manul_transform,
                                                                   test_transform=manul_transform,
                                                                   batch_size=BATCH_SIZE)
    print(f"Number of batches of size {BATCH_SIZE} in 10 percent training data: {len(train_dataloader_10_percent)}")
    print(f"Number of batches of size {BATCH_SIZE} in 20 percent training data: {len(train_dataloader_20_percent)}")
    print(f"Number of batches of size {BATCH_SIZE} in testing data: {len(test_dataloader)} (all experiments will use the same test set)")
    print(f"Number of classes: {len(class_names)}, class names: {class_names}")
    return train_dataloader_10_percent, train_dataloader_20_percent, test_dataloader

def main():
    create_dataset()

    num_epochs = [5, 10]
    models = ["effnetb0", "effnetb2"]

    train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                         "data_20_percent": train_dataloader_20_percent}

    set_seeds()

    experiment_number = 0

    for dataloader_name, train_dataloader in train_dataloaders.items():
        for epochs in num_epochs:
            for model_name in models:
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                if model_name == "effnetb0":
                    model = create_effnetb0()
                else:
                    model = create_effnetb2()

                loss_fn = torch.nn.CrossEntropyLoss()
                optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

                writer = create_writer(experienment_name=dataloader_name,
                                       model_name=model_name,
                                       extra=f"epochs_{epochs}")

                train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      loss_fn=loss_fn,
                      optim=optim,
                      epochs=epochs,
                      writer=writer)

                save_path = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                torch.save(obj=model.state_dict(), f=save_path)
                print("-"*50 + "\n")

if __name__ == "__main__":
    main()