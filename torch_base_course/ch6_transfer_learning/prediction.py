from typing import List, Tuple

import PIL.Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from PIL import Image

from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_top5_wrong_images(model: torch.nn.Module,
                        images_path: List[Path],
                        class_names: List[str],
                        image_size: Tuple[int, int]=(224, 224),
                        transform: torchvision.transforms=None,
                        device:torch.device="cpu"):
    test_pred_list = []
    for image_path in images_path:
        pred_dict = {}
        pred_dict["img_path"] = image_path
        class_name = image_path.parent.stem
        pred_dict['class_name'] = class_name
        img = Image.open(image_path)
        # print(f'input img shape and type: {img.size}, {type(img)}')
        # class_name = image_path.strip().split("\\")[-2]
        # print(f'img class name: {class_name}')
        if transform:
            img_transform = transform
        else:
            img_transform = transforms.Compose([
                transforms.RandomCrop(size=image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        img_transformed = img_transform(img).unsqueeze(dim=0).to(device)
        # print(f'After transforms, img shape: {img_transformed.shape}')

        model.to(device)
        model.eval()
        with torch.inference_mode():
            logits = model(img_transformed)

            scores = torch.softmax(logits, dim=1)

            prediction = torch.argmax(scores, dim=1)
            pred_class = class_names[prediction.cpu()]
            pred_dict["pred_prob"] = scores.max().cpu()
            pred_dict["pred_class"] = pred_class

            if pred_class != class_name:
                test_pred_list.append(pred_dict)
    print(f'before sorted: {test_pred_list}')
    test_pred_list.sort(key=lambda item: item["pred_prob"], reverse=True)
    print(f'after sorted: {test_pred_list}')

    for i, item in enumerate(test_pred_list):
        if i < 5:
            img = PIL.Image.open(item["img_path"])
            true_label = item["class_name"]
            pred_prob = item["pred_prob"]
            pred_class = item["pred_class"]
            plt.figure()
            plt.imshow(img)
            plt.title(f'True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}')
            plt.axis(False)
            plt.show()

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int]=(224, 224),
                        transform: torchvision.transforms=None,
                        device:torch.device="cpu"):
    img = Image.open(image_path)
    print(f'input img shape and type: {img.size}, {type(img)}')

    if transform:
        img_transform = transform
    else:
        img_transform = transforms.Compose([
            transforms.RandomCrop(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    img_transformed = img_transform(img).unsqueeze(dim=0).to(device)
    print(f'After transforms, img shape: {img_transformed.shape}')

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(img_transformed)

        socres = torch.softmax(logits, dim=1)

        prediction = torch.argmax(socres, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[prediction]} | Prob: {socres.max():.3f}')
    plt.axis(False)
    plt.show()

def batch_test():
    test_dir = Path("../ch4_dataset/data/pizza_steak_sushi/test")
    test_imgs = list(test_dir.glob("*/*.jpg"))
    print(f'test images number: {len(test_imgs)}')

    test_img_samples = random.sample(test_imgs, k=3)

    # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=3)
    )
    model.load_state_dict(torch.load("./efficientnet_b0_epoch10.pth"))
    # print(next(model.parameters()))

    # test_img_samples = ["../ch5_going_modular/04-pizza-dad.jpeg"]

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for img in test_img_samples:
            pred_and_plot_image(model=model,
                                image_path=img,
                                class_names=['pizza', 'steak', 'sushi'])


if __name__=="__main__":
    test_dir = Path("../ch4_dataset/data/pizza_steak_sushi/test")
    test_imgs = list(test_dir.glob("*/*.jpg"))
    print(f'test images number: {len(test_imgs)}, {test_imgs[0]}')

    # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=3)
    )
    model.load_state_dict(torch.load("./efficientnet_b0_epoch10.pth"))
    pred_and_plot_top5_wrong_images(model, test_imgs, class_names=['pizza', 'steak', 'sushi'])