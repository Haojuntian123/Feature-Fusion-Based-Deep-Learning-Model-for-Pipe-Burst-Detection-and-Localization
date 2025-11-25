# coding=utf-8
# @software:    PyCharm
# @name:        CNN_model.py
# @description: Defines a Convolutional Neural Network (CNN) model for pipe burst warning and location identification.
#               Model architecture includes convolutional layers, max pooling, and a classification head.

import torch
import torch.nn as nn
from config import IN_CHANNEL, HIDDEN_CHANNEL1, HIDDEN_CHANNEL2, OUT_CHANNEL, HIDDEN_FEATURE, OUT_FEATURES

class MyCNN(nn.Module):
    """
    Custom CNN model for detecting and locating pipe bursts.
    Args:
        in_channel (int): Number of input channels.
        hidden_channel1 (int): Number of channels in the first hidden layer.
        hidden_channel2 (int): Number of channels in the second hidden layer.
        out_channel (int): Output channels after convolutional blocks.
        hidden_feature (int): Feature size in the first linear layer.
        out_features (int): Output feature size (number of classes).
    """
    def __init__(self, in_channel, hidden_channel1, hidden_channel2, out_channel, hidden_feature, out_features):
        super(MyCNN, self).__init__()
        # Convolutional blocks with max pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel1, kernel_size=3, padding=2),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=hidden_channel1, out_channels=hidden_channel2, kernel_size=(5, 3), padding=2),
            nn.MaxPool2d(kernel_size=(5, 3)),
            nn.Conv2d(in_channels=hidden_channel2, out_channels=out_channel, kernel_size=(3, 3), padding=2)
        )
        self.adp = nn.AdaptiveMaxPool2d(output_size=1)  # Adaptive max pooling to fixed size
        # Classification head with linear layers and softmax
        self.classification = nn.Sequential(
            nn.Linear(in_features=out_channel, out_features=hidden_feature),
            nn.ReLU(),
            nn.Linear(in_features=hidden_feature, out_features=out_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the CNN model.
        Args:
            x (Tensor): Input tensor of shape (B, S, F); unsqueezed to (B, C, S, F).
        Returns:
            Tensor: Output probabilities of shape (B, out_features).
        """
        x.unsqueeze_(dim=1)  # Add channel dimension to tensor
        x = self.cnn(x)       # Apply convolutional layers
        x = self.adp(x)       # Apply adaptive pooling
        x = self.classification(x.squeeze())  # Classification and softmax
        return x

if __name__ == '__main__':
    # Load parameters from config.py and initialize model
    model = MyCNN(IN_CHANNEL, HIDDEN_CHANNEL1, HIDDEN_CHANNEL2, OUT_CHANNEL, HIDDEN_FEATURE, OUT_FEATURES)
    print("Model initialized with configuration parameters.")
