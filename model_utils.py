import os
import torch
from torch import nn, optim
from torchvision import models

def build_model(arch='vgg16', hidden_units=512, learning_rate=0.001, device='cpu'):
    """
    Build a model according to the specified arguments
    """
    architectures = {
        'vgg16': (models.vgg16, 25088),
        'resnet18': (models.resnet18, 512),
        'resnet50': (models.resnet50, models.resnet50(pretrained=True).fc.in_features),
        'googlenet': (models.googlenet, models.googlenet(pretrained=True).fc.in_features)
    }
    
    if arch not in architectures:
        raise ValueError(f"Unsupported architecture: {arch}. Try choosing from resnet18, resnet50, vgg16, or googlenet")

    model_fn, input_size = architectures[arch]
    model = model_fn(pretrained=True)
    
    # Freeze the feature extraction parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Define the new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Set the model classifier/fc
    if arch in ['vgg16']:
        model.classifier = classifier
    else:
        model.fc = classifier
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if arch == 'vgg16' else model.fc.parameters(), lr=learning_rate)
    
    # Move model to device 
    model.to(device)
    return model, criterion, optimizer

def save_checkpoint(model, optimizer, class_to_idx, save_path='checkpoint_.pth', arch='vgg16', hidden_units=512, learning_rate=0.001, epochs=5):
    """
    Save model checkpoint after training
    """
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint successfully saved to path: {save_path}")

def load_checkpoint(filepath, device='cpu'):
    """
    Load a pretrained model from a checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at path: '{filepath}'")
      
    # Load checkpoint to device 
    checkpoint = torch.load(filepath, map_location=device)
    
    # Rebuild the model with our previous function passing the checkpoint keys
    model, _, _ = build_model(
        arch=checkpoint.get('arch', 'vgg16'),
        hidden_units=checkpoint.get('hidden_units', 512),
        learning_rate=checkpoint.get('learning_rate', 0.001),
        device=device
    )
    
    # Load saved parameters
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print(f"Model successfully loaded from {filepath}")
    
    return model
