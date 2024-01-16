import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from torchinfo import summary

from pathlib import Path

import sys
sys.path.append("../ch5_going_modular")
from ch5_going_modular import dataset_setup, engine
from utils import *
from vit_model import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

img_path = Path("C:\Workspace\develop\pytorch\\torch_base_course\ch7_experiment_tracking\dataset")
print(f'Data path is {img_path}')
train_dir = img_path / "train"
test_dir = img_path / "test"

IMG_SIZE = 224
BATCH_SIZE = 32
PATCH_SIZE = 16

train_dataloader, test_dataloader = None, None
class_names = None
def create_dataloader():
    manul_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    global train_dataloader, test_dataloader
    global class_names
    train_dataloader, test_dataloader, class_names = dataset_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=manul_transform,
        test_transform=manul_transform,
        batch_size=BATCH_SIZE
        )
    print(train_dataloader, class_names)

    imgs, labels = next(iter(train_dataloader))
    print(imgs[0].shape, labels[0])
    # plot_img(imgs[0], class_names[labels[0]])

    # image_permuted = imgs[0].permute(1,2,0)
    # plt.figure(figsize=(PATCH_SIZE, PATCH_SIZE))
    # plt.imshow(image_permuted[:PATCH_SIZE, : , :])
    # plt.show()

    # num_patches = IMG_SIZE // PATCH_SIZE
    # assert IMG_SIZE % PATCH_SIZE == 0
    # print(f'patch number is {num_patches}')
    # fig, axs = plt.subplots(nrows= num_patches,
    #                         ncols=num_patches,
    #                         figsize=(num_patches, num_patches),
    #                         sharex=True,
    #                         sharey=True)
    # for i, path_height in enumerate(range(0, IMG_SIZE, PATCH_SIZE)):
    #     for j , path_width in enumerate(range(0, IMG_SIZE, PATCH_SIZE)):
    #         axs[i, j].imshow(image_permuted[path_height: path_height + PATCH_SIZE,
    #                          path_width: path_width + PATCH_SIZE, :])
    #         axs[i, j].set_ylabel(i + 1,
    #                              rotation="horizontal",
    #                              horizontalalignment="right",
    #                              verticalalignment="center")
    #         axs[i, j].set_xlabel(j + 1)
    #         axs[i, j].set_xticks([])
    #         axs[i, j].set_yticks([])
    #         axs[i, j].label_outer()
    # fig.suptitle(f"{class_names[labels[0]]} -> Patchified", fontsize=16)
    # plt.show()

def train_vit():

    create_dataloader()

    set_seed()

    model = ViT(class_num=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(),
                             betas=(0.9, 0.999),
                             weight_decay=0.3,
                             lr=3e-3)

    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optim=optim,
                           epochs=10)

def transfer_learning():
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    auto_transforms = weights.transforms()
    print(f'auto transforms is {auto_transforms}')

    train_dataloader, test_dataloader, class_names = dataset_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=auto_transforms,
        test_transform=auto_transforms,
        batch_size=BATCH_SIZE
    )

    model = torchvision.models.vit_b_16(weights=weights).to(device)

    # print(f'pretrained model is {model}')

    for param in model.parameters():
        param.requires_grad = False

    set_seed()

    model.heads = nn.Linear(in_features=model.hidden_dim, out_features=len(class_names)).to(device)

    # summary(model=model,
    #         input_size=(1, 3, 224, 224),  # (batch_size, num_patches, embedding_dimension)
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(),
                             betas=(0.9, 0.999),
                             weight_decay=0.3,
                             lr=3e-3)

    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optim=optim,
                           epochs=1)

    save_model(model=model, target_dir="models",
               model_name="08_pretrained_vit_feature_extractor.pth")

    plot_loss_curves(results)

def predict():
    test_img = "./04-pizza-dad.jpeg"

    manul_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    model = torchvision.models.vit_b_16()
    model.heads = nn.Linear(in_features=model.hidden_dim, out_features=3)
    model.load_state_dict(torch.load("models/08_pretrained_vit_feature_extractor.pth"))



    pred_and_plot_image(model=model,
                        image_path=test_img,
                        transform=manul_transform)

def test():
    conv2d = nn.Conv2d(in_channels=3, out_channels=768,
                       kernel_size=PATCH_SIZE, stride=PATCH_SIZE,
                       padding=0)
    input_imgs, labels = next(iter(train_dataloader))
    img, label = input_imgs[0], labels[0]
    print(f'input img shape: {img.shape}, label is {label}')

    img_out_of_conv = conv2d(img.unsqueeze(dim=0))
    print(f'img shape out of convd: {img_out_of_conv.shape}')

    # import random
    # random_indexes = random.sample(range(0, 758), k=5)
    # fig, axs = plt.subplots(nrows=1, ncols=len(random_indexes), figsize=(12,12))
    # for i, idx in enumerate(random_indexes):
    #     image_conv_feature_map = img_out_of_conv.squeeze()[idx, : , :]
    #     axs[i].imshow(image_conv_feature_map.detach().numpy())
    #     axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # plt.show()

    flat = nn.Flatten(start_dim=2, end_dim=3)
    img_after_flat = flat(img_out_of_conv).permute(0, 2, 1)
    print(f'img shape after flatten: {img_after_flat.shape}')

if __name__ == "__main__":
    # create_dataloader()
    #
    # test()

    # train_vit()

    # transfer_learning()

    predict()