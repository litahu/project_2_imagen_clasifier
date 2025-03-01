import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

def load_data(data_dir):
    """
    Load train, validation, and test datasets while applying transformations
    """
    # Define data directories 
    dirs = {x: os.path.join(data_dir, x) for x in ['train', 'valid', 'test']}

    # Define transforms 
    transform = {
        'train': transforms.Compose([
            transforms.RandomRotation(35),
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    datasets = {x: ImageFolder(dirs[x], transform=transform['train' if x == 'train' else 'eval']) for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=64 if x == 'train' else 32, shuffle=True if x == 'train' else False) for x in ['train', 'valid', 'test']}
    
    # Get class_to_idx mapping 
    class_to_idx = datasets['train'].class_to_idx  

    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], class_to_idx

def process_image(image_path):
    """
    Process an image preparing it for use in the app model
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File '{image_path}' not found!")

    image = Image.open(image_path).convert("RGB")

    # Define transformation as expected by the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Return a tensor image
    image = transform(image)

    return image
