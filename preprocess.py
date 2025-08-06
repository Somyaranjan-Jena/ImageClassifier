import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    # Define transforms for training (with augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation and testing (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transforms)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    data_dir = "data"
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir)
    print(f"Classes: {classes}")
    # Verify batch shapes
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}, Labels shape: {labels.shape}")



# continued preprocess.py 
def visualize_batch(loader, classes):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()  # CHW to HWC
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)  # Ensure pixel values are valid
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    data_dir = "data"
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir)
    print(f"Classes: {classes}")
    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {images.shape}, Labels shape: {labels.shape}")
    visualize_batch(train_loader, classes)  # Add this line