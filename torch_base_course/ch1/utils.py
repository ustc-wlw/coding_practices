import matplotlib.pyplot as plt

def plot_predictions(train_data=None, train_label=None,
                     test_data=None, test_label=None,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_label, c='b', s=4, label="Training data")
    plt.scatter(test_data, test_label, c='g', s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()