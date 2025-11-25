# coding=utf-8
# @name:        ImproveFeatures.py
# @software:    PyCharm
# @description: Flow-Attention based Feature Enhancement Modules

import torch
import torch.nn as nn
import config  # Import model configuration module


class LocationAwareProjection(nn.Module):
    """
    Position-sensitive feature enhancement module
    Architecture:
      Feature Reshape → Adjacency Weighting → Position Encoding
    """

    def __init__(self, feature_dim, sequence_len):
        """
        Initialize localization module
        Args:
            feature_dim: Feature dimension per sensor
            sequence_len: Temporal sequence length
        """
        super(LocationAwareProjection, self).__init__()
        num_sensors = config.NUM_PRESSURE_SENSORS

        # Position encoding parameters
        self.position_encoding = nn.Parameter(
            torch.Tensor(feature_dim * sequence_len, num_sensors),
            requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.position_encoding)

        # Register hydraulic graph (non-trainable)
        self.register_buffer(
            'adjacency_matrix',
            torch.tensor(config.SENSOR_ADJACENCY_MATRIX, dtype=torch.float32)
        )

        # Feature projection layer
        self.feature_projection = nn.Linear(num_sensors, feature_dim)

    def forward(self, sensor_data):
        """
        Spatial feature enhancement
        Args:
            sensor_data: [batch, sensors, features]
        Returns:
            Enhanced features with positional encoding
        """
        batch_size, num_sensors, feature_dim = sensor_data.size()
        flattened = sensor_data.reshape(batch_size, -1)

        # Stage 1: Spatial attention weighting
        location_attention = flattened.mm(self.position_encoding)

        # Stage 2: Graph topology transformation
        topological_features = location_attention.mm(self.adjacency_matrix)

        # Stage 3: Feature space mapping
        return self.feature_projection(topological_features).unsqueeze(1) * sensor_data


class FlowAttentionGate(nn.Module):
    """
    Flow-Attention Feature Selection Module
    Architecture:
      Flow Encoder → Attention Pooling → Feature Selection
    """

    def __init__(self, flow_sensors, hidden_dim):
        """
        Initialize flow attention mechanism
        Args:
            flow_sensors: Number of flow measurement points
            hidden_dim: Hidden dimension size
        """
        super(FlowAttentionGate, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(flow_sensors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.PRESSURE_FEATURE_DIM),
            nn.Sigmoid()
        )

    def forward(self, pressure_data, flow_data):
        """
        Flow-guided feature selection
        Args:
            pressure_data: [batch, sensors, features]
            flow_data: [batch, flow_sensors]
        Returns:
            Flow-attentive pressure features
        """
        # Generate attention weights
        feature_weights = self.attention_network(flow_data).unsqueeze(1)

        # Apply flow-based attention
        return pressure_data * feature_weights


class TopologyFlowIntegration(nn.Module):
    """
    Hydraulic Topology-Aware Feature Fusion
    Architecture:
      Flow-Sensor Mapping → Feature Encoding → Topology-Weighted Fusion
    """

    def __init__(self, hidden_dim):
        """
        Initialize topology-aware fusion module
        Args:
            hidden_dim: Encoder hidden dimension
        """
        super(TopologyFlowIntegration, self).__init__()
        num_sensors = config.NUM_PRESSURE_SENSORS

        # Feature encoding network
        self.encoding_network = nn.Sequential(
            nn.Linear(num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sensors),
            nn.Sigmoid()
        )

        # Register hydraulic relationship matrix
        self.register_buffer(
            'hydraulic_matrix',
            torch.tensor(config.HYDRAULIC_CORRELATION_MATRIX, dtype=torch.float32)
        )

    def forward(self, pressure_data, flow_data):
        """
        Topology-aware feature fusion
        Args:
            pressure_data: [batch, sensors, features]
            flow_data: [batch, flow_sensors]
        Returns:
            Fused features with hydraulic awareness
        """
        # Hydraulic correlation projection
        topology_features = flow_data.mm(self.hydraulic_matrix)

        # Feature attention encoding
        attention_weights = self.encoding_network(topology_features).unsqueeze(1)

        # Apply topology-aware weighting
        return pressure_data * attention_weights
