import argparse

import torch
import torchvision

from model import CNN
from train import Trainer
from evaluation import eval

def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=20, help="training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--bs', type=int, default=128, help="batch size")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = CNN()

    # get datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # train
    trainer = Trainer(model=model)
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    acc  = eval(model, test_loader)
    print("Accuracy on test.py set: %.2f" % acc)
    return


if __name__ == "__main__":
    main()
