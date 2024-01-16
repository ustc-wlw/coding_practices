import torch.cuda
import torchvision
from torchinfo import summary
import torch.nn as nn

class_names = ['pizza', 'steak', 'sushi']

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seeds(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_effnetb0():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
    ).to(device)

    # summary(model=model, input_size=(32, 3, 224, 224),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         )

    model.name = "effnetb0"

    return model

def create_effnetb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds()

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408, out_features=len(class_names), bias=True)
    ).to(device)

    # summary(model=model, input_size=(32, 3, 224, 224),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         )

    model.name = "effnetb2"

    return model

if __name__ == "__main__":
    # create_effnetb0()

    model = create_effnetb2()
    # print(model.name)