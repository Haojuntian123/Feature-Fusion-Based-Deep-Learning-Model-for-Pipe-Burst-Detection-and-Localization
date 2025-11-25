# coding=utf-8
# @name:        operate_inp.py
# @software:    PyCharm
# @description: Configuration and optimization of hydraulic network models

import numpy as np
import wntr
from scipy.interpolate import interp1d
import config  # Import configuration settings


class OperateModel:
    """
    Configures hydraulic model parameters and performs pattern interpolation.

    Attributes:
        wn (wntr.WaterNetworkModel): Hydraulic network model
        old_step (int): Original pattern timestep (seconds)
        new_step (int): Interpolated pattern timestep (seconds)
    """

    def __init__(self, wn, old_step=3600, new_step=60):
        """
        Args:
            wn: wntr WaterNetworkModel instance
            old_step: Original pattern timestep in seconds
            new_step: Target pattern timestep in seconds
        """
        self.wn = wn
        self.old_step = old_step
        self.new_step = new_step

    def _config_hydraulic_params(self):
        """Configures pressure-driven demand (PDD) parameters."""
        hyd_options = self.wn.options.hydraulic
        hyd_options.demand_model = 'PDD'
        hyd_options.required_pressure = config.REQ_PRESSURE
        hyd_options.minimum_pressure = config.MIN_PRESSURE
        hyd_options.pressure_exponent = config.PRESSURE_EXPONENT

    def _config_time_params(self):
        """Configures simulation time parameters."""
        time_options = self.wn.options.time
        time_options.duration = config.SIM_DURATION
        time_options.hydraulic_timestep = config.HYD_TIMESTEP
        time_options.pattern_timestep = config.PATTERN_TIMESTEP
        time_options.report_timestep = config.REPORT_TIMESTEP

    def _interpolate_patterns(self):
        """
        Performs quadratic interpolation on demand patterns with optional noise.
        """
        for pattern_name in self.wn.pattern_name_list:
            pattern = self.wn.get_pattern(pattern_name)
            x = np.arange(0, (len(pattern) + 1) * self.old_step, self.old_step)
            y = list(pattern.multipliers)
            y.append(pattern.multipliers[0])  # Wrap-around
            f = interp1d(x, y, kind='quadratic')

            # Create new time grid
            nx = np.arange(0, len(pattern) * self.old_step, self.new_step)
            ny = f(nx)

            # Apply optional disturbance
            if config.ADD_PATTERN_NOISE:
                noise = 1.0 - np.random.normal(
                    loc=0,
                    scale=config.NOISE_STD,
                    size=ny.shape
                )
                ny *= noise

            pattern.multipliers = ny.tolist()

    def configure_and_save(self):
        """Main configuration pipeline with optional file saving."""
        # Configure hydraulic parameters
        self._config_hydraulic_params()

        # Configure time parameters
        self._config_time_params()

        # Interpolate patterns
        self._interpolate_patterns()

        # Save modified model
        self.wn.write_inpfile(
            filename=config.MODIFIED_MODEL_PATH,
            version=2.2
        )
        print(f"Modified model saved to: {config.MODIFIED_MODEL_PATH}")


class CheckModel:
    """
    (Reserved class for structural model calibration)
    Purpose: Reduces discrepancy between model and real-world network structure.
    """

    def __init__(self, wn):
        self.wn = wn
        # Placeholder for structural calibration methods
        # Implementation pending based on network-specific requirements


if __name__ == '__main__':
    # Load hydraulic model from config
    wn = wntr.network.WaterNetworkModel(config.MODEL_FILE)

    # Configure model parameters and save
    OperateModel(
        wn,
        old_step=config.ORIG_PATTERN_STEP,
        new_step=config.NEW_PATTERN_STEP
    ).configure_and_save()
