import torch
import torchvision

print(f'torch version: {torch.__version__}')
print(f'torchvison version: {torchvision.__version__}')

import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

import sys
sys.path.append("C:\Workspace\develop\pytorch\torch_base_course")
from ch5_going_modular import dataset_setup, engine

import matplotlib.pyplot as plt
from pathlib import Path

from engine import train

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seeds(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    data_dir = Path("../ch4_dataset/data/pizza_steak_sushi")
    train_data_path = data_dir / "train"
    test_data_path = data_dir / "test"

    manul_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    train_dataloader, test_dataloader, class_names = dataset_setup.create_dataloaders(
        train_dir=train_data_path,
        test_dir=test_data_path,
        train_transform=manul_transform,
        test_transform=manul_transform,
        batch_size=32)

    # print(f'train : {train_dataloader},test samples {test_dataloader}, {class_names}')

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transform = weights.transforms()
    print(f'model input transforms is {auto_transform}')
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    # print(model)

    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    ).to(device)

    # summary(model=model, input_size=(32,3,224,224),
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #         col_width=20)

    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

    results = train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optim=optim,
                    loss_fn=loss,
                    epochs=5)


if __name__ == "__main__":
    main()