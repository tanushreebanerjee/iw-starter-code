# main.py

import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from code.models.simple_cnn import SimpleCNN
from code.losses.custom_loss import CustomLoss
from code.datasets.custom_dataset import CustomDataset
from code.utils.trainer import Trainer

# Training Script
def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=10):
    trainer = Trainer(model, criterion, optimizer, device)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    val_accuracy = trainer.evaluate(val_loader)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("Training complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for a custom CNN model.")
    
    parser.add_argument("--train_data_path", type=str, default="data/train_data.pth",
                        help="Path to the training data file.")
    parser.add_argument("--train_targets_path", type=str, default="data/train_targets.pth",
                        help="Path to the training targets file.")
    
    parser.add_argument("--val_data_path", type=str, default="data/val_data.pth",
                        help="Path to the validation data file.")
    parser.add_argument("--val_targets_path", type=str, default="data/val_targets.pth",
                        help="Path to the validation targets file.")
    
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes in the classification problem.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for optimizer.")
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["cross_entropy", "custom"],
                        help="Loss function to use for training (cross_entropy or custom).")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training (cuda or cpu).")
    
    parser.add_argument("--dataset", type=str, default="custom", choices=["custom", "cifar10"],
                        help="Dataset to use for training (custom or cifar10).")
    return parser.parse_args()


def train(args):
    args = parse_arguments()

    # Load data
    train_data = torch.load(args.train_data_path)
    train_targets = torch.load(args.train_targets_path)
    val_data = torch.load(args.val_data_path)
    val_targets = torch.load(args.val_targets_path)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets
    if args.dataset == "custom":
        train_dataset = CustomDataset(train_data, train_targets, transform=transform)
        val_dataset = CustomDataset(val_data, val_targets, transform=transform)
    elif args.dataset == "cifar10":
        # Load CIFAR-10 dataset
        assert args.num_classes == 10, "Number of classes for CIFAR-10 should be 10."
        train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
        val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = SimpleCNN(num_classes=args.num_classes)
    if args.loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "custom":
        criterion = CustomLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, args.device, args.num_epochs)


def train_torchvision_dataset(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.dataset == "cifar10":
        train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
        val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    elif args.dataset == "custom":
        train_data = torch.load(args.train_data_path)
        train_targets = torch.load(args.train_targets_path)
        val_data = torch.load(args.val_data_path)
        val_targets = torch.load(args.val_targets_path)

        train_dataset = CustomDataset(train_data, train_targets, transform=transform)
        val_dataset = CustomDataset(val_data, val_targets, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = SimpleCNN(num_classes=10)  # Assuming CIFAR-10 has 10 classes
    criterion = CustomLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, args.device, args.num_epochs)


def main():
    args = parse_arguments()
    train(args)

if __name__ == "__main__":
    main()
