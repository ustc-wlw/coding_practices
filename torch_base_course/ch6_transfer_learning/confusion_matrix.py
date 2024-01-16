import sys
# print(f'currrent work path: {sys.path}')
sys.path.append("../ch5_going_modular")

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from ch5_going_modular import dataset_setup

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def main():
    train_dir = "../ch4_dataset/data/pizza_steak_sushi/train"
    test_dir = "../ch4_dataset/data/pizza_steak_sushi/test"

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

    model = torchvision.models.efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=3)
    )
    model.load_state_dict(torch.load("./efficientnet_b0_epoch10.pth"))

    y_preds = []
    y_labels = []

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for batch_imgs, labels in test_dataloader:
            batch_imgs = batch_imgs.to(device)

            logits = model(batch_imgs)
            print(f'logits shape: {logits.shape}')
            predicts = torch.softmax(logits, dim=1)
            preds = torch.argmax(predicts, dim=1).cpu()
            y_preds.append(preds)

            y_labels.append(labels)
        y_preds_tensor = torch.cat(y_preds)
        y_labels_tensor = torch.cat(y_labels)
        print(f'y_preds_tensor shape: {y_preds_tensor.shape}, {y_labels_tensor.shape}')

        confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
        confmat_tensor = confmat(preds=y_preds_tensor, target=y_labels_tensor)

        fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                        class_names=class_names,
                                        figsize=(10, 7))
        plt.show()

if __name__=="__main__":
    main()