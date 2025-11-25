# coding=utf-8
# @name:        dataset.py
# @software:    PyCharm
# @description: Pipeline dataset generation and graph structure construction

import wntr
import numpy as np
import pandas as pd
import networkx as nx
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import config  # Import configuration settings


class CreateGraph:
    """
    Construct topological graph representations from hydraulic network models.
    Attributes:
        wn (wntr.WaterNetworkModel): Hydraulic network model instance.
    """

    def __init__(self, wn):
        """
        Args:
            wn: wntr WaterNetworkModel instance
        """
        self.wn = wn

    def reverse_pipes(self):
        """
        Identify pipes with reverse flow direction under baseline conditions.

        Returns:
            list: Pipe IDs with reverse flow
        """
        time = self.wn.options.time.duration
        self.wn.options.time.duration = 0
        sim = wntr.sim.EpanetSimulator(self.wn)
        result = sim.run_sim()
        self.wn.options.time.duration = time
        self.wn.reset_initial_values()
        pipes = result.link['flowrate'].iloc[0, :][result.link['flowrate'].iloc[0, :] < 0].keys()
        return pipes

    def wsn_graph(self, remove_reservoirs=True):
        """
        Construct directed graph matching hydraulic flow directions.

        Args:
            remove_reservoirs: Exclude reservoirs from graph (default=True)

        Returns:
            nx.DiGraph: Directed graph representation
            np.array: Adjacency matrix (float32)
        """
        pipes = self.reverse_pipes()
        g = self.wn.get_graph()
        g_copy = self.wn.get_graph().copy()

        # Correct edge directions for reverse-flow pipes
        for edge in g_copy.edges:
            for pipe in pipes:
                if pipe in edge:
                    g.remove_edge(edge[0], edge[1])
                    g.add_edge(edge[1], edge[0])

        # Remove reservoir nodes if specified
        if remove_reservoirs:
            for i in range(self.wn.num_nodes - self.wn.num_reservoirs, self.wn.num_nodes):
                g.remove_node(self.wn.node_name_list[i])
        return g, np.asarray(nx.adjacency_matrix(g).todense()).astype(np.float32)

    def knn_graph(self):
        """
        Construct KNN graph with 2-hop directional connectivity.

        Returns:
            nx.DiGraph: Directed KNN graph
            np.array: Adjacency matrix (float32)
        """
        g, adj = self.wsn_graph()
        nodes = list(g.nodes)

        for n, node in enumerate(nodes):
            # Downstream connectivity (2 hops)
            for v in g.successors(node):
                adj[n, nodes.index(v)] = 1
                for sv in g.successors(v):
                    adj[n, nodes.index(sv)] = 1
            # Upstream connectivity (2 hops)
            for u in g.predecessors(node):
                adj[n, nodes.index(u)] = 1
                for su in g.predecessors(u):
                    adj[n, nodes.index(su)] = 1

        return g, adj.astype(np.float32)

    def monitor_graph(self, monitors):
        """
        Construct adjacency matrix for sensor nodes considering flow paths.

        Args:
            monitors: List of sensor node IDs

        Returns:
            np.array: Sensor adjacency matrix (float32)
            list: Node coordinates [(x1,y1), ...]
        """
        g, _ = self.wsn_graph()
        pos = nx.get_node_attributes(g, 'pos')
        # Get monitor coordinates
        ps = [pos[node] for node in monitors]

        # Build adjacency based on direct flow paths
        graph = np.zeros((len(monitors), len(monitors)))
        for i, src in enumerate(monitors):
            for j, target in enumerate(monitors):
                if nx.has_path(g, src, target):
                    path = nx.dijkstra_path(g, src, target)
                    if len(set(monitors) & set(path)) <= 2:  # Direct path
                        graph[i, j] = 1
        return graph.astype(np.float32), ps

    def cal_g(self, flow_sensors, pressure_sensors):
        """
        Compute connectivity matrix between flow and pressure sensors.

        Args:
            flow_sensors: List of flow sensor node IDs
            pressure_sensors: List of pressure sensor node IDs

        Returns:
            np.array: Connectivity matrix (N_flow x M_pressure)
        """
        graph, _ = self.wsn_graph(remove_reservoirs=False)
        connectivity = np.zeros((len(flow_sensors), len(pressure_sensors)), dtype=np.float32)

        for i, flow_node in enumerate(flow_sensors):
            for j, pressure_node in enumerate(pressure_sensors):
                if nx.has_path(graph, flow_node, pressure_node):
                    connectivity[i, j] = 1
        return connectivity

    def index2id(self, indexs):
        """Convert numerical indices to node IDs."""
        return [self.wn.node_name_list[i] for i in indexs]


class MyLocationData(Dataset):
    """Dataset for pipe burst localization."""

    def __init__(self, data_file, label_file, need_time_len=60,
                 time_series_len=60, left_offset=0, right_offset=0,
                 is_norm=False):
        """
        Args:
            data_file: Path to pressure data CSV
            label_file: Path to label CSV
            need_time_len: Required time steps per sample
            time_series_len: Total available time steps
            left_offset, right_offset: Time window shifts
            is_norm: Apply standardization if True
        """
        self.data = pd.read_csv(data_file, header=None).values
        if is_norm:
            self.data = StandardScaler().fit_transform(self.data)
        self.labels = pd.read_csv(label_file, header=None, index_col=0).values
        self.need_len = need_time_len
        self.total_len = time_series_len
        self.left_shift = left_offset
        self.right_shift = right_offset

    def __getitem__(self, idx):
        """Extract time window and corresponding label."""
        start_idx = idx * self.total_len
        x = self.data[start_idx:start_idx + self.total_len, :]
        y = self.labels[idx][0] - 1  # Adjust for 0-indexing

        # Calculate temporal window
        base_start = (self.total_len - self.need_len) // 2
        start = base_start + self.left_shift
        end = self.total_len - base_start + self.right_shift
        x = x[start:end, :].astype(np.float32)

        return x, y.astype(np.int64)

    def __len__(self):
        return self.labels.shape[0]


# ============== Implementation of other Dataset classes remains unchanged ==============
# MyLocationBigData, MyLocationDataQ, MyLocationBigDataQ,
# MyAlarmData, MyAlarmDataQ preserved with original functionality
# =======================================================================================

if __name__ == '__main__':
    # Load configurations
    wn = wntr.network.WaterNetworkModel(config.MODEL_FILE)
    model = CreateGraph(wn)

    # Compute sensor connectivity
    conn_matrix = model.cal_g(
        flow_sensors=config.FLOW_SENSORS,
        pressure_sensors=config.PRESSURE_MONITORS
    )
    print("Sensor connectivity matrix:\n", conn_matrix)
