from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 1000

X,y = make_circles(n_samples=n_samples, noise=0.03,
                   random_state=42)

print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

torch.manual_seed(42)

# plt.scatter(x=X[:,0], y=X[:,1],
#             c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32).unsqueeze(dim=1)
print(f'X shape: {X.shape}, y shape: {y.shape}')

print(X[:2])

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(input)))))

model = Classifier(2, 10).to(device)
print(f'model is {model}')

loss_fn = torch.nn.BCEWithLogitsLoss()

optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

def accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    assert predictions.shape == labels.shape
    return torch.eq(predictions,labels).sum().item() / predictions.shape[0] * 100

epoches = 1000

X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

for epoch in range(epoches):
    model.train()

    y_logits = model(X_train)
    # print(f'output logits shape: {y_logits.shape}')
    y_pred_label = torch.round(torch.sigmoid(y_logits))
    # print(f'output label shape: {y_pred_label.shape}')
    # print(y_logits[:5])
    # print(y_pred_label[:5])

    loss = loss_fn(y_logits, y_train)
    acc = accuracy(y_pred_label, y_train)

    optim.zero_grad()

    loss.backward()

    optim.step()

    model.eval()

    with torch.inference_mode():
        y_logits = model(X_test)
        y_pred_label = torch.round(torch.sigmoid(y_logits))

        loss_test = loss_fn(y_logits, y_test)
        acc_test = accuracy(y_pred_label, y_test)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss} | Accuracy: {acc:.2f}')

def plt_prediction():
    from helper_functions import plot_predictions, plot_decision_boundary
    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

plt_prediction()