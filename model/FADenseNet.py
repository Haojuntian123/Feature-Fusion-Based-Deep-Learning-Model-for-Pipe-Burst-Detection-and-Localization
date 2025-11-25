# coding=utf-8
# @name:        FADenseNet.py
# @software:    PyCharm
# @description: Feature-Attention Dense Network for Time Series Classification

import torch
import torch.nn as nn
import config  # Import configuration module


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Layer for feature recalibration
    """

    def __init__(self, channels):
        """
        Args:
            channels (int): Number of input channels
        """
        super(ChannelAttention, self).__init__()
        reduction = config.SE_REDUCTION_RATIO
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation with feature recalibration
        Args:
            x (Tensor): Input features [batch, channels]
        Returns:
            Tensor: Recalibrated features
        """
        return x * self.fc(x)


class DenseBlock(nn.Module):
    """
    Densely Connected Feature Extraction Block
    """

    def __init__(self, in_dim, out_dim):
        """
        Initialize feature extraction block
        Args:
            in_dim (int): Input feature dimension
            out_dim (int): Output feature dimension
        """
        super(DenseBlock, self).__init__()

        # Get hidden layer dimensions from config
        hidden_sizes = config.FA_HIDDEN_LAYER_DIMS

        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], out_dim),
            nn.Dropout(config.DROPOUT_RATE)
        )

    def forward(self, x):
        """
        Feature transformation pipeline
        Args:
            x (Tensor): Input features
        Returns:
            Tensor: Processed features
        """
        return self.layers(x)


class FeatureAttentionDenseNet(nn.Module):
    """
    Feature-Attention Dense Network core architecture
    """

    def __init__(self, in_features, out_classes):
        """
        Initialize network components
        Args:
            in_features (int): Input feature dimension per time step
            out_classes (int): Number of output classes
        """
        super(FeatureAttentionDenseNet, self).__init__()

        # Calculate sequence-related dimensions
        time_steps = config.SEQUENCE_LENGTH
        feature_dim = config.INITIAL_FEATURE_DIM * time_steps

        # Network architecture configuration
        self.dense_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        # Input projection
        self.input_proj = nn.Linear(
            in_features * time_steps,
            feature_dim
        )

        # Block configurations from config
        block_config = config.DENSE_BLOCK_CONFIGURATIONS

        # Create dense blocks and attention layers
        for i, cfg in enumerate(block_config):
            layer_out = cfg['output_multiplier'] * time_steps

            self.dense_layers.append(
                DenseBlock(
                    feature_dim if i == 0 else self.dense_layers[-1].layers[-4].out_features,
                    layer_out
                )
            )

            if cfg['use_attention']:
                self.attention_layers.append(
                    ChannelAttention(layer_out)
                )

        # Final classification layer
        self.classifier = nn.Sequential(
            DenseBlock(
                self.dense_layers[-1].layers[-4].out_features,
                config.FINAL_HIDDEN_DIM * time_steps
            ),
            nn.Linear(
                config.FINAL_HIDDEN_DIM * time_steps,
                out_classes
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward propagation through network
        Args:
            x (Tensor): Input data [batch, time_steps, features]
        Returns:
            Tensor: Class probabilities [batch, classes]
        """
        # Flatten temporal dimension
        flat = x.view(x.size(0), -1)
        features = self.input_proj(flat)

        # Residual connections container
        residuals = [features]

        # Process through dense blocks
        for i, dense_layer in enumerate(self.dense_layers):
            # Forward through block
            out = dense_layer(residuals[-1])

            # Apply attention if configured
            if i < len(self.attention_layers):
                out = self.attention_layers[i](out)

            # Dense connection
            residuals.append(out + sum(residuals))

        # Final classification
        return self.classifier(residuals[-1])
