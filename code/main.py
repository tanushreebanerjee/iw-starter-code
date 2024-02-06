import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

    print("Training complete.")

# Sample usage
if __name__ == "__main__":
    # Dummy data (replace with your actual data)
    train_data = torch.randn(100, 3, 32, 32)
    train_targets = torch.randint(0, 10, (100,))
    val_data = torch.randn(20, 3, 32, 32)
    val_targets = torch.randint(0, 10, (20,))

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets
    train_dataset = CustomDataset(train_data, train_targets, transform=transform)
    val_dataset = CustomDataset(val_data, val_targets, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # Initialize model, criterion, optimizer
    model = SimpleCNN(num_classes=10)
    criterion = CustomLoss()  # Using custom loss function
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, device)