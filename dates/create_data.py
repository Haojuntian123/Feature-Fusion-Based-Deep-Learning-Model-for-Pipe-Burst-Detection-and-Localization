# coding=utf-8
# @name:        create_data.py
# @software:    PyCharm
# @description: Generates hydraulic network datasets including normal operation data, burst event simulations, and sensitivity matrices.

import numpy as np
import pandas as pd
import wntr
import os
from multiprocessing import Pool
from config import (  # Import configuration settings
    INP_FILE, BURST_DATA_DIR,
    NORMAL_PRESSURE_FILE, NORMAL_FLOW_FILE,
    BURST_LEVEL, DISCHARGE_COEFF,
    DEFAULT_BURST_START, BURST_INTERVAL
)


class CreateData:
    """
    Generates simulation datasets for hydraulic network analysis.

    Key functionalities:
    1. normal_data: Generates pressure and flow data under normal conditions.
    2. pipe_burst: Simulates pipe bursts at junctions and records pressure/flow changes.
    3. sensitive: Computes sensitivity matrix to quantify burst impacts.
    """

    def __init__(
            self,
            wn,
            burst_level=BURST_LEVEL,
            duration=None,
            burst_interval=BURST_INTERVAL,
            discharge_coeff=DISCHARGE_COEFF):
        """
        Args:
            wn: wntr WaterNetworkModel instance
            burst_level: array-like, burst severity levels (fraction of pipe diameter)
            duration: int, simulation duration in seconds (default=model duration)
            burst_interval: int, time between consecutive bursts in seconds
            discharge_coeff: float, discharge coefficient for burst orifice equation
        """
        self.wn = wn
        if duration:
            self.duration = duration
        else:
            self.duration = self.wn.options.time.duration
        self.burst_level = np.array(burst_level)
        self.burst_interval = burst_interval
        self.discharge_coeff = discharge_coeff

    def normal_data(
            self,
            is_save=True,
            Pfname=NORMAL_PRESSURE_FILE,
            Qfname=NORMAL_FLOW_FILE):
        """
        Simulates and saves pressure/flow data under normal operating conditions.

        Args:
            is_save: bool, whether to save results to CSV
            Pfname: str, output path for pressure data
            Qfname: str, output path for flow data

        Returns:
            pdata: DataFrame, node pressures (rows: timesteps, cols: junctions)
            qdata: DataFrame, reservoir flows (rows: timesteps, cols: reservoirs)
        """
        sim = wntr.sim.WNTRSimulator(self.wn)
        result = sim.run_sim()
        self.wn.reset_initial_values()
        pdata = result.node['pressure'].iloc[:-1, :-3]
        qdata = -result.node['demand'].iloc[:-1, -3:]
        if is_save:
            # Ensure output directories exist
            os.makedirs(os.path.dirname(Pfname), exist_ok=True)
            pdata.to_csv(Pfname)
            qdata.to_csv(Qfname)
        return pdata, qdata

    def _get_diameter(self, node):
        """Retrieves maximum pipe diameter connected to a junction node."""
        diameters = [self.wn.get_link(link).diameter
                     for link in self.wn.get_links_for_node(node)]
        return max(diameters) if diameters else 0.0

    def _burst(self, node, area, start_time):
        """
        Simulates a single burst event at specified node and start time.

        Args:
            node: str, junction ID
            area: float, burst orifice area (m²)
            start_time: int, burst start time in seconds

        Returns:
            p_data: DataFrame, pressure data during burst (61 timesteps)
            q_data: Series, leak flow during burst
            Q: DataFrame, reservoir flows during burst
        """
        self.wn.options.time.duration = start_time + self.burst_interval
        junction = self.wn.get_node(node)
        junction.remove_leak(self.wn)
        junction.add_leak(
            self.wn,
            area=area,
            discharge_coeff=self.discharge_coeff,
            start_time=start_time,
            end_time=start_time + self.burst_interval
        )
        sim = wntr.sim.WNTRSimulator(self.wn)
        result = sim.run_sim()
        self.wn.reset_initial_values()
        p_data = result.node['pressure'].iloc[-61:-1, :-3]
        q_data = result.node['leak_demand'][node].iloc[-61:-1]
        Q = -result.node['demand'].iloc[-61:-1, -3:]
        return p_data, q_data, Q

    def pipe_burst(self, start_time=DEFAULT_BURST_START, nodes=None):
        """
        Generates burst simulation datasets for specified nodes and burst levels.

        Output files:
        - P_junction_X.XX.csv: Pressure data for burst level X.XX
        - Q_junction_X.XX.csv: Leak flow data
        - Q_all_junction_X.XX.csv: Reservoir flows
        - junction_burst_area.csv: Burst areas used (m²)

        Args:
            start_time: int, initial burst start time (seconds)
            nodes: list of junction IDs to simulate (default=all junctions)
        """
        node_list = nodes or self.wn.junction_name_list
        # Create output directory if missing
        os.makedirs(BURST_DATA_DIR, exist_ok=True)

        for node in node_list:
            diameter = self._get_diameter(node)
            areas = np.pi * (diameter ** 2) / 4 * self.burst_level
            for i, area in enumerate(areas):
                start = start_time
                P_datas = pd.DataFrame()
                Q_datas = pd.DataFrame()
                Q_all = pd.DataFrame()
                # Simulate bursts at regular intervals over duration
                while start <= (self.duration - self.burst_interval):
                    p_data, q_data, q = self._burst(node, area, start)
                    P_datas = pd.concat([P_datas, p_data])
                    Q_datas = pd.concat([Q_datas, q_data])
                    Q_all = pd.concat([Q_all, q])
                    start += self.burst_interval

                # Save datasets with burst severity in filename
                severity = '%.2f' % self.burst_level[i]
                P_datas.to_csv(f"{BURST_DATA_DIR}/P_{node}_{severity}.csv")
                Q_datas.to_csv(f"{BURST_DATA_DIR}/Q_{node}_{severity}.csv")
                Q_all.to_csv(f"{BURST_DATA_DIR}/Q_all_{node}_{severity}.csv")
                print(f"{node}_{severity} completed")
            # Save burst area values for current node
            np.savetxt(
                f"{BURST_DATA_DIR}/{node}_burst_area.csv",
                areas,
                fmt='%.6f',
                delimiter=','
            )
        self.wn.reset_initial_values()

    def sensitive(self, bursts, start_time=DEFAULT_BURST_START):
        """
        Computes sensitivity matrix: pressure change per unit leak flow.

        Args:
            bursts: float, burst severity level to analyze
            start_time: int, burst start time (seconds)
        """
        try:
            normal = pd.read_csv(NORMAL_PRESSURE_FILE, header=0, index_col=0)
        except FileNotFoundError:
            normal, _ = self.normal_data(is_save=True)
        pressure = normal.loc[start_time].values
        node_list = self.wn.junction_name_list
        sen_matrix = np.zeros((0, len(node_list)))

        for node in node_list:
            diameter = self._get_diameter(node)
            area = np.pi * (diameter ** 2) / 4 * bursts
            df, bq, _ = self._burst(node, area, start_time)
            mid_idx = int(df.shape[0] / 2)
            delta_p = pressure - df.values[mid_idx, :]
            # Normalize by pressure change at burst node
            sen = delta_p / (pressure[node_list.index(node)] - df.values[mid_idx, node_list.index(node)])
            sen_matrix = np.vstack([sen_matrix, sen])

        # Save sensitivity matrix with parameters in filename
        output_path = f"./datas/sensitive_{bursts}_{start_time // 3600}.csv"
        np.savetxt(output_path, sen_matrix, fmt='%.4f', delimiter=',')
        self.wn.reset_initial_values()


def mulcreatedata(wn, burst_level, nodes):
    """
    Multiprocessing helper function for burst dataset generation.

    Args:
        wn: wntr WaterNetworkModel instance
        burst_level: array-like, burst severity levels
        nodes: list of junction IDs to process
    """
    model = CreateData(wn, burst_level)
    model.pipe_burst(nodes=nodes)


if __name__ == '__main__':
    # Initialize hydraulic model from config
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    # Generate normal condition datasets
    datamodel = CreateData(wn, BURST_LEVEL)
    datamodel.normal_data()

    # Uncomment below to run full burst simulations
    # datamodel.pipe_burst()

    # Uncomment below to run sensitivity analysis
    # datamodel.sensitive(burst_level=1.0, start_time=12*3600)

    # Uncomment for multiprocessing burst generation
    '''
    node_list = wn.junction_name_list
    chunk_size = len(node_list) // 10 + 1
    pool = Pool(10)

    for i in range(10):
        n = node_list[i*chunk_size: (i+1)*chunk_size]
        pool.apply_async(mulcreatedata, args=(wn, BURST_LEVEL, n))
        print(f'Process {i} started: {len(n)} nodes')

    pool.close()
    pool.join()
    print("All burst datasets generated")
    '''
