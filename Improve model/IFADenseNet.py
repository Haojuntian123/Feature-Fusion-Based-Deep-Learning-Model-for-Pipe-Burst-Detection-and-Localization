# coding=utf-8
# @name:        IFADenseNet.py
# @software:    PyCharm
# @description: Improved Feature-Attention Network for Hydraulic Burst Localization

import torch
import torch.nn as nn
import config  # Import model configuration module
from model.improve_location.ImproveFeatures import LAQ


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation layer for feature recalibration
    """

    def __init__(self, channels, reduction_ratio=config.ATTENTION_REDUCTION):
        """
        Args:
            channels: Input feature channels
            reduction_ratio: Channel reduction ratio
        """
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Feature recalibration with channel-wise attention
        Args:
            x: Input tensor [batch, features]
        Returns:
            Calibrated feature tensor
        """
        return x * self.fc(x)


class FeatureBlock(nn.Module):
    """
    Dense Feature Extraction Block with non-linear transformation
    """

    def __init__(self, in_features, expand_factor=config.EXPAND_FACTOR):
        """
        Args:
            in_features: Input feature dimension
            expand_factor: Hidden dimension expansion ratio
        """
        super(FeatureBlock, self).__init__()

        # Dynamic hidden layer sizes based on expansion factor
        hidden_dim1 = int(in_features * expand_factor)
        hidden_dim2 = int(in_features * expand_factor * 0.75)

        self.features = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, in_features),
            nn.Dropout(config.DROPOUT_RATE)
        )

    def forward(self, x):
        """
        Dense feature transformation
        Args:
            x: Input feature tensor [batch, features]
        Returns:
            Transformed feature tensor
        """
        return self.features(x)


class IFANet(nn.Module):
    """
    Improved Feature-Attention Network for burst localization
    """

    def __init__(self):
        """
        Initialize network with configurable parameters
        """
        super(IFANet, self).__init__()

        # Dynamic feature dimensions based on sequence length
        base_features = config.SEQUENCE_LENGTH * config.NUM_PRESSURE_SENSORS

        # Feature fusion module
        self.feature_fusion = LAQ(
            config.FLOW_SENSOR_GRAPH,
            config.NUM_PRESSURE_SENSORS,
            config.FUSION_OUTPUT_DIM
        )

        # Network components
        self.input_projection = nn.Linear(
            base_features,
            config.PROJECTED_FEATURES
        )

        # Block configurations
        self.feature_blocks = nn.ModuleList([
            FeatureBlock(config.PROJECTED_FEATURES)
            for _ in range(config.NUM_BLOCKS)
        ])

        self.attention_layers = nn.ModuleList([
            ChannelAttention(config.PROJECTED_FEATURES)
            for _ in range(config.NUM_BLOCKS - 1)
        ])

        # Output module
        self.output_block = nn.Sequential(
            FeatureBlock(config.PROJECTED_FEATURES, expand_factor=3.0),
            nn.Linear(config.PROJECTED_FEATURES, config.NUM_LEAK_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, data_input):
        """
        Forward propagation pipeline
        Args:
            data_input: Tuple (pressure_data, flow_data)
              pressure_data: [batch, sensors, time_steps]
              flow_data: [batch, flow_sensors]
        Returns:
            Location prediction probabilities [batch, classes]
        """
        pressure_data, flow_data = data_input

        # Stage 1: Feature fusion
        fused = self.feature_fusion(pressure_data, flow_data)
        reshaped = fused.view(fused.size(0), -1)

        # Stage 2: Feature projection
        features = self.input_projection(reshaped)

        # Stage 3: Cascade processing
        residual = features
        for block, attention in zip(self.feature_blocks[:-1], self.attention_layers):
            output = block(residual)
            output = attention(output)
            residual = residual + output

        # Stage 4: Final prediction
        return self.output_block(residual)
