# coding=utf-8
# @name:        IRNN.py
# @software:    PyCharm
# @description: Integrated Hydraulic Anomaly Detection Network

import torch
import torch.nn as nn
from model.feature_enhance import TopologyFlowIntegration
import config  # Centralized configuration

# Deterministic setup for reproducibility
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)
device = torch.device(config.DEVICE)


class HydraulicRNN(nn.Module):
    """
    Integrated Hydraulic Anomaly Detection Network
    Architecture:
      Feature Enhancement → Temporal Modeling → Spatial Localization
    """

    def __init__(self):
        """
        Initialize diagnostic network components
        """
        super(HydraulicRNN, self).__init__()

        # Feature enhancement module
        self.feature_enhancer = TopologyFlowIntegration(
            config.LAQ_HIDDEN_DIM
        )

        # Temporal modeling module
        self.temporal_model = nn.GRU(
            input_size=config.NUM_PRESSURE_FEATURES,
            hidden_size=config.GRU_HIDDEN_DIM,
            num_layers=config.GRU_NUM_LAYERS,
            batch_first=True,
            dropout=config.GRU_DROPOUT
        )

        # Diagnostic classification module
        self.localization_classifier = nn.Sequential(
            nn.Linear(config.GRU_HIDDEN_DIM, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CLASSIFIER_HIDDEN_DIM,
                      config.NUM_LEAKAGE_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, sensor_data):
        """
        End-to-end hydraulic diagnosis
        Args:
            sensor_data: Tuple (pressure_data, flow_data)
              pressure_data: [batch, sequence, features]
              flow_data: [batch, num_flow_sensors]
        Returns:
            Class probabilities for leakage localization
        """
        pressure_data, flow_data = sensor_data

        # Feature enhancement with hydraulic topology
        enhanced_features = self.feature_enhancer(
            pressure_data, flow_data
        )

        # Temporal feature extraction
        temporal_features, _ = self.temporal_model(enhanced_features)

        # Terminal feature pooling (last time-step)
        terminal_state = temporal_features[:, -1, :]

        # Leakage localization
        return self.localization_classifier(terminal_state)
