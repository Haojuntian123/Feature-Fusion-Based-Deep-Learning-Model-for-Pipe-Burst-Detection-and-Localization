# coding=utf-8
# @name:        ICNN.py
# @software:    PyCharm
# @description: Hydraulic Convolutional Diagnostic Network

import torch
import torch.nn as nn
from model.feature_enhance import TopologyFlowIntegration
import config  # Centralized configuration module


class HydraulicCNN(nn.Module):
    """
    Convolutional Network for Hydraulic Anomaly Diagnosis
    Architecture:
      Feature Enhancement → Spatio-temporal Convolution → Localization
    """

    def __init__(self):
        """
        Initialize convolutional diagnostic model
        """
        super(HydraulicCNN, self).__init__()

        # Topology-aware feature enhancement
        self.feature_enhancer = TopologyFlowIntegration(
            config.CNN_LAQ_HIDDEN_DIM
        )

        # Spatio-temporal convolution module
        self.convolution_module = nn.Sequential(
            nn.Conv2d(
                in_channels=config.CNN_IN_CHANNELS,
                out_channels=config.CNN_CHANNEL1,
                kernel_size=3,
                padding=config.CNN_PADDING
            ),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(
                in_channels=config.CNN_CHANNEL1,
                out_channels=config.CNN_CHANNEL2,
                kernel_size=(5, 3),
                padding=config.CNN_PADDING
            ),
            nn.MaxPool2d(kernel_size=(5, 3)),
            nn.Conv2d(
                in_channels=config.CNN_CHANNEL2,
                out_channels=config.CNN_OUT_CHANNELS,
                kernel_size=(3, 3),
                padding=config.CNN_PADDING
            )
        )

        # Dimensionality reduction layer
        self.feature_pooling = nn.AdaptiveMaxPool2d(output_size=1)

        # Diagnostic classifier
        self.diagnostic_classifier = nn.Sequential(
            nn.Linear(config.CNN_OUT_CHANNELS, config.CNN_HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(config.CNN_HIDDEN_FEATURES, config.NUM_LEAKAGE_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, sensor_data):
        """
        Pipeline for hydraulic anomaly diagnosis
        Args:
            sensor_data: Tuple (pressure_tensor, flow_vector)
              pressure_tensor: [batch, sequence_len, features]
              flow_vector: [batch, num_flow_sensors]
        Returns:
            Leakage location probabilities [batch, num_classes]
        """
        pressure_data, flow_data = sensor_data

        # Feature enhancement
        enhanced = self.feature_enhancer(pressure_data, flow_data)

        # Add channel dimension: [batch, sequence, features] => [batch, channels, sequence, features]
        convolved = enhanced.unsqueeze(1)

        # Spatio-temporal feature extraction
        convolved = self.convolution_module(convolved)

        # Feature compaction
        compressed = self.feature_pooling(convolved).squeeze()

        # Leakage localization
        return self.diagnostic_classifier(compressed)
