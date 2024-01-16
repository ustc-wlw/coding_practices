import os
import torch
import dataset_setup, engine, model_builder, utils

from torchvision import transforms
import argparse

# NUM_EPOCHS = 1
# BATCH_SIZE = 32
# HIDDEN_DIM = 10
# LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):

    ## 1 data
    train_dir = args.train_dir
    test_dir = args.test_dir

    NUM_EPOCHS = args.epochs_num
    BATCH_SIZE = args.batch_size
    HIDDEN_DIM = args.hidden_size
    LEARNING_RATE = args.learning_rate

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()]
    )

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()]
    )

    train_dataloader, test_dataloader, class_names = dataset_setup.create_dataloaders(train_dir=train_dir,
                                                                        test_dir=test_dir,
                                                                        train_transform=train_transform,
                                                                        test_transform=test_transform,
                                                                        batch_size=BATCH_SIZE)
    print(f'class names: {class_names}')

    imgs, labels = next(iter(test_dataloader))
    print(f'img shape: {imgs.shape}, batch label len: {len(labels)}, labels: {labels}')

    ## 2 model
    model = model_builder.TinyVGG(input_shape=3, hidden_dim=HIDDEN_DIM, output_shape=len(class_names))
    model.to(device)

    ## 3 loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    ## 4 train and test
    engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                 optim=optim, loss_fn=loss_fn, epochs=NUM_EPOCHS, device=device)

    ## 5 save model
    utils.save_model(model, save_dir="./models", model_name="tinyVGG.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='training Arguments',
                                     allow_abbrev=False)
    parser.add_argument("--train-dir", type=str, default="../ch4_dataset/data/pizza_steak_sushi/train")
    parser.add_argument("--test-dir", type=str, default="../ch4_dataset/data/pizza_steak_sushi/test")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-num", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=10)
    args = parser.parse_args()
    print(f'args: {args}')
    main(args)