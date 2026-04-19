from torchvision.models import resnet50, ResNet50_Weights

import torch
from torch import nn

def create_model(num_classes):
    """Creates a ResNet50 model with a custom classifier for naira note classification"""
    #Load pre trained resnet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    #Replace final layer for 5 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model