# coding=utf-8
# @name:        RNN_model.py
# @software:    PyCharm
# @description: Defines a Gated Recurrent Unit (GRU) model for pipe burst warning and location identification.
#               The model processes time-series data and outputs classification probabilities.

import torch
import torch.nn as nn
from config import SEED, RNN_IN_FEATURES, GRU_HIDDEN, LINEAR_HIDDEN, RNN_OUT_FEATURES, RNN_NUM_LAYERS

# Set random seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyRNN(nn.Module):
    """
    RNN model using GRU layers for pipe burst detection and localization.
    Args:
        in_features (int): Number of input features per time step.
        gru_hidden (int): Size of hidden state in GRU layers.
        linear_hidden (int): Size of hidden layer in classification head.
        out_features (int): Number of output classes (locations).
        num_layers (int): Number of stacked GRU layers.
    """
    def __init__(self, in_features, gru_hidden, linear_hidden, out_features, num_layers):
        super(MyRNN, self).__init__()
        # GRU layer for sequence processing
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True
        )
        # Classification head with linear layers and softmax
        self.classification = nn.Sequential(
            nn.Linear(gru_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the RNN model.
        Args:
            x (Tensor): Input tensor of shape (B, S, F) where
                        B: batch size, S: sequence length, F: features
        Returns:
            Tensor: Output probabilities of shape (B, out_features).
        """
        # Process sequence through GRU layer
        gru_x, _ = self.gru(x)
        # Extract last time step output
        lx = gru_x[:, -1, :]
        # Classification output
        x = self.classification(lx)
        return x

if __name__ == '__main__':
    # Initialize model using configuration parameters
    model = MyRNN(
        in_features=RNN_IN_FEATURES,
        gru_hidden=GRU_HIDDEN,
        linear_hidden=LINEAR_HIDDEN,
        out_features=RNN_OUT_FEATURES,
        num_layers=RNN_NUM_LAYERS
    )
    print("RNN model initialized with configuration parameters.")
