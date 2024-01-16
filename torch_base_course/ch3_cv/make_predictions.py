import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import os

import matplotlib.pyplot as plt

random.seed(42)

from minist_dataset import FashionMNISTModelV2, get_dataset, test_dataloader

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

device = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = "models/03_pytorch_computer_vision_model_2.pth"

model_loaded = FashionMNISTModelV2(input_shape=1,
                                   hidden_units=10,
                                   output_shape=10)
if os.path.exists(model_save_path):
    model_loaded.load_state_dict(torch.load(model_save_path))
else:
    print('checkpoint path is not valid!!!')

print(model_loaded)

train_data, test_data = get_dataset()
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

def make_predictions(model: torch.nn.Module,
                     imgs: list,
                     device: torch.device=torch.device("cpu")):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for img in imgs:
            logits = model(torch.unsqueeze(img.to(device), dim=0))
            # print(f'logits shape: {logits.shape}')
            pred_prob = torch.softmax(logits.squeeze(), dim=0)
            # print(f'pred_prob shape: {pred_prob.shape}')

            pred_probs.append(pred_prob.cpu())

    predictions = torch.stack(pred_probs)
    print(f'result prediction shape: {predictions.shape}')
    return predictions

prediction_probs = make_predictions(model_loaded, test_samples)

prediction_labels = torch.argmax(prediction_probs, dim=1)
print(f'pred_labels: {prediction_labels}\n samples labels: {test_labels}')

def plt_predictions():
    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i + 1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[prediction_labels[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]]

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r")  # red text if wrong
        plt.axis(False)
    plt.show()

# plt_predictions()

def plt_confusion_matrix(y_pred_tensor):
    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix

    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                             target=test_data.targets)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
        class_names=class_names, # turn the row and column labels into class names
        figsize=(10, 7)
    )
    plt.show()

def make_predictions(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device=torch.device("cpu")):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            logits = model(X.to(device))
            # print(f'logits shape: {logits.shape}')
            pred_class = torch.softmax(logits, dim=1).argmax(dim=1)

            pred_probs.append(pred_class.cpu())

    predictions = torch.cat(pred_probs)
    print(f'result prediction shape: {predictions.shape}')
    return predictions

predictions = make_predictions(model_loaded, test_dataloader)
plt_confusion_matrix(predictions)