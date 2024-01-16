import os
import sys
# print(f'currrent work path: {sys.path}')
sys.path.append("../ch5_going_modular")

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from torchinfo import summary
from timeit import default_timer as timer

from ch5_going_modular import dataset_setup, engine


from pathlib import Path
import random

from helper_functions import plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

def main():
    img_dir = Path("../ch4_dataset/data/pizza_steak_sushi")
    assert img_dir.exists(), "data not exists!!!"

    train_dir = img_dir / "train"
    test_dir = img_dir / "test"

    manul_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    # print(f'weights: {weights}')
    auto_transform = weights.transforms()
    # print(f'auto transforms: {auto_transform}')

    ### setup dataset
    train_dataloader, test_dataloader, class_names = dataset_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=auto_transform,
        test_transform=auto_transform, batch_size=32
    )
    # print(train_dataloader, test_dataloader, class_names)
    images, labels = next(iter(test_dataloader))
    print(f'batch img shape: {images.shape}, label shape: {labels.shape}')


    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    # print(model)

    # summary(model=model, input_size=(32, 3, 224, 224),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    ).to(device)

    # summary(model=model, input_size=(32, 3, 224, 224),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)

    ### training
    start = timer()

    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optim=optim,
                           loss_fn=loss_fn,
                           epochs=10)
    # print(f'train results: {results}')

    end = timer()
    print(f'[INFO] Total training time: {end - start:.3f} seconds')

    model_save_path = "./efficientnet_b0_epoch10.pth"
    torch.save(obj=model.state_dict(), f=model_save_path)
    print(f'save model successful at {model_save_path}')

    plot_loss_curves(results)
    plt.show()

if __name__=="__main__":
    main()