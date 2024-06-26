import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


def get_dataset_loaders():
    train_dataset = datasets.MNIST(
        root="./mnist", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="./mnist", train=False, transform=transforms.ToTensor()
    )

    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=0,
        batch_size=64,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
