import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
# Code for creating a spiral dataset from CS231n
import numpy as np

def accuracy_fn(pred, label):
    assert pred.shape == label.shape
    correct = torch.eq(pred, label).sum().item()
    return correct / label.shape[0] * 100

def gen_data():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j
    # lets visualize the data
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    return X, y

SEED = 42

X, y = gen_data()

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.LongTensor)
print(f'X shape: {X.shape}, y shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear3 = nn.Linear(in_features=hidden_size, out_features=num_class)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.linear3(self.relu(self.linear2(self.relu(self.linear1(input)))))


model = Classifier(2, 10, 3).to(device)
# print(f'model is {model.state_dict()}')

loss_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

epoches = 1000

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for epoch in range(epoches):
    model.train()

    y_logits = model(X_train)

    y_scores = torch.softmax(y_logits, dim=1)
    # print(f'ouput logits shape: {y_logits.shape}')

    y_labels = torch.argmax(y_scores, dim=1).squeeze()
    # print(f'output predictions labels shape: {y_labels.shape}, {y_labels[:5]}')

    loss = loss_fn(y_logits, y_train)
    acc_train = accuracy_fn(y_labels, y_train)

    optim.zero_grad()

    loss.backward()

    optim.step()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_test)

        y_scores = torch.softmax(y_logits, dim=1)
        # print(f'test ouput logits shape: {y_logits.shape}')

        y_labels = torch.argmax(y_scores, dim=1).squeeze()
        # print(f'test output predictions labels shape: {y_labels.shape}, {y_labels[:5]}')

        loss_test = loss_fn(y_logits, y_test)
        acc_test = accuracy_fn(y_labels, y_test)

    if epoch % 10 == 0:
        print(f'Epoch {epoch} | Traing loss {loss} | Training Accuracy {acc_train} | Test loss {loss_test} | Test Accuracy {acc_test}')

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