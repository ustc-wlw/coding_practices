
import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from timeit import default_timer as timer

from helper_functions import accuracy_fn, print_train_time

from pathlib import Path
def get_dataset():
    train_data = datasets.FashionMNIST(
        root='data', train=True, download=True, transform=ToTensor(), target_transform=None
    )

    test_data = datasets.FashionMNIST(
        root='data', train=False, download=True, transform=ToTensor()
    )
    return train_data, test_data

train_data, test_data = get_dataset()

print(f'train_data len: {len(train_data)}')
print(f'test_data len: {len(test_data)}')

img, label = train_data[0]
print(f'input img shape: {img.shape}, label : {label}')

class_names = train_data.classes
print(f'class names: {class_names}')


def plt_img(img, label):
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.show()

# img, label = train_data[0]
# plt_img(img, label)

torch.manual_seed(42)

def batch_img_plt():
    fig = plt.figure(figsize=(9,9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        # print(f'random idx is {random_idx}')
        img, label = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)

    plt.show()

# batch_img_plt()

train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

print(f'Length of train dataloader: {len(train_dataloader)} of batch size 32')
print(f'Length of test dataloader: {len(test_dataloader)} of batch size 32')
# print(next(iter(train_dataloader)))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

class FashionMNISTModel_V0(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModel_V0(28 * 28, hidden_dim=10, output_shape=len(class_names)).to(device)
print("model_0 is ", model_0)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model_0.parameters(), lr=0.1)

epochs = 3

def train_eval(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optim: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device=torch.device("cpu")):
    train_time_start = timer()

    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}\n ........................')
        train_loss = 0
        model_0.train()
        for batch_id, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            logits = model_0(X)
            # print(f'logits shape: {logits.shape}')
            loss =  loss_fn(logits, y)
            train_loss += loss

            optim.zero_grad()

            loss.backward()

            optim.step()

            if batch_id % 400 == 0:
                print(f'Looked at {batch_id * len(X)} / {len(train_dataloader.dataset)} samples')

        train_loss = train_loss / len(train_dataloader)

        model_0.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for batch_id, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                logits = model_0(X)
                # print(f'logits shape: {logits.shape}')
                loss = loss_fn(logits, y)
                test_loss += loss
                test_acc += accuracy_fn(torch.argmax(logits, dim=1), y)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f'\nTraining loss: {train_loss:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc:2f}% \n')

    train_time_end = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start, end=train_time_end,
                                                device=str(next(model_0.parameters()).device))
    return total_train_time_model_0

torch.manual_seed(42)

def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=torch.device("cpu")):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)

            loss += loss_fn(logits, y)
            acc += accuracy_fn(torch.argmax(logits, dim=1), y)
        loss /= len(dataloader)
        acc /= len(dataloader)

    return {"model_name":model.__class__.__name__,
            "model_loss":loss.item(),
            "model_acc":acc}

# model_0_results = eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)
# print(model_0_results)

class FashionMNISTModel_V1(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)

model_1 = FashionMNISTModel_V1(28 * 28, 10, len(class_names)).to(device)
print('model_1 is: ', model_1)

optim_1 = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optim: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device=torch.device("cpu")):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch_id, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss =  loss_fn(logits, y)
        train_loss += loss
        train_acc += accuracy_fn(torch.argmax(logits, dim=1), y)

        optim.zero_grad()

        loss.backward()

        optim.step()

        if batch_id % 400 == 0:
            print(f'Looked at {batch_id * len(X)} / {len(train_dataloader.dataset)} samples')

    train_loss = train_loss / len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               accuracy_fn,
               device: torch.device=torch.device("cpu")):
        test_loss = 0
        test_acc = 0
        model.to(device)
        model.eval()
        with torch.inference_mode():
            for batch_id, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                test_loss += loss
                test_acc += accuracy_fn(torch.argmax(logits, dim=1), y)

            test_loss /= len(dataloader)
            test_acc /= len(dataloader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def train_model(model: torch.nn.Module,
                optim: torch.optim.Optimizer):
    train_time_start = timer()

    epochs = 3
    for epoch in range(epochs):
        print(f'Epoch: {epoch}\n ...............')
        train_step(model, train_dataloader, loss_fn, optim, accuracy_fn)
        test_step(model, test_dataloader, loss_fn, accuracy_fn)

    train_time_end = timer()
    total_train_time = print_train_time(start=train_time_start, end=train_time_end, device=device)
    return total_train_time

# train_model(model_1, optim_1)
#
# model_1_results = eval_model(model=model_1,
#                              dataloader=test_dataloader,
#                              loss_fn=loss_fn,
#                              accuracy_fn=accuracy_fn)
# print(model_1_results)

# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
print('model_2 is: ', model_2)

# optim_2 = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

# train_model(model=model_2, optim=optim_2)
#
# model_2_results = eval_model(model=model_2,
#                              dataloader=test_dataloader,
#                              loss_fn=loss_fn,
#                              accuracy_fn=accuracy_fn)
# print(model_2_results)

def save_model():
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = "03_pytorch_computer_vision_model_2.pth"
    model_save_path = model_dir / model_name
    print(f'model save path: {model_save_path}')

    torch.save(obj=model_2.state_dict(), f=model_save_path)
