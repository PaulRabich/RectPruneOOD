import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from rect_prune_ood.lib.models.resnet_cifar_react import resnet50_cifar



# Define a custom dataset for CIFAR-10
class CIFAR10CustomDataset(Dataset):
    def __init__(self, data_dir, files, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        # Load data from batches
        for file in files:
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.data.append(batch['data'])
                self.labels.extend(batch['labels'])
        
        # Convert to numpy arrays
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).astype("float32")
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(torch.tensor(image))  # Transform expects tensors
        return image, label

# Set paths to the dataset directory and batch files
data_dir = './cifar-10-batches-py'  # Adjust path as needed
train_files = [f"data_batch_{i}" for i in range(1, 6)]  # Batches 1-5 for training
test_file = "test_batch"  # Test batch

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL image
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 mean/std
])

# Create train and test datasets
train_dataset = CIFAR10CustomDataset(data_dir, train_files, transform=transform)
test_dataset = CIFAR10CustomDataset(data_dir, [test_file], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer
model = resnet50_cifar(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_resnet50_cifar10.pth')
        print("Model weights saved.")

print("Training complete.")

# Load the saved model weights
model.load_state_dict(torch.load('best_resnet50_cifar10.pth'))
model.eval()
print("Model weights loaded and ready for inference.")