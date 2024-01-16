import torch
from torch import nn
from utils import plot_predictions

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')

weight = 0.7
bias = 0.3

X = torch.arange(start=0, end=1, step=0.02).unsqueeze(dim=1)
print(f'X shape: {X.shape}')
Y = X * weight + bias

print(f'X shape: {X.shape}, Y shape: {Y.shape}')
# print(f'{X[:10]}, {Y[:10]}')

train_split = int(len(X) * 0.8)
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

print(len(X_train), len(Y_train), len(X_test), len(Y_test))

# plot_predictions(train_data=X_train, train_label=Y_train,
#                  test_data=X_test, test_label=Y_test)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, input):
        return self.linear_layer(input)

torch.manual_seed(42)
model = LinearRegressionModel()
print(f'model is {model} \n model state_dict: {model.state_dict()}')

print(next(model.parameters()).device)
model.to(device)
print(next(model.parameters()).device)

loss_fn = nn.L1Loss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

epoch_num = 1000

def train():
    for i in range(epoch_num):
        model.train()

        optim.zero_grad()

        predictions = model(X_train.to(device))
        loss = loss_fn(Y_train.to(device), predictions)

        loss.backward()
        optim.step()

        model.eval()
        with torch.inference_mode():
            Y_pred = model(X_test.to(device))
            loss_test = loss_fn(Y_pred, Y_test.to(device))

        if i % 100 == 0:
            print(f"Epoch: {i} | Training loss: {loss} | Test loss: {loss_test}")

    # Find our model's learned parameters
    from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html
    print("The model learned the following values for weights and bias:")
    pprint(model.state_dict())
    print("\nAnd the original values for weights and bias are:")
    print(f"weights: {weight}, bias: {bias}")

# def save_model():
from pathlib import Path

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_name = "01_pytorch_workflow_model.pth"
model_save_path = model_dir / model_name

print(f'Saving model to {model_save_path}')
# torch.save(obj=model.state_dict(), f=model_save_path)

model_new = LinearRegressionModel()
model_new.load_state_dict(torch.load(model_save_path))
print(f'model_new is {model_new}')

model_new.eval()
with torch.inference_mode():
    predictions = model_new(X_test.to(device))

    plot_predictions(train_data=X_train,
                     train_label=Y_train,
                     test_data=X_test,
                     test_label=Y_test,
                     predictions=predictions)