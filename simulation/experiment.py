#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for implementing a forward simulation experiment
with different wave equation solvers. Handling of data saving
and data loading.}

{
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
}
"""

# -------- IMPORTS --------
# Built-in modules
import datetime
import pathlib
import json
import pickle

# Other modules
import numpy as np

# Own modules
from config.logger import Logger
from utility.plotting import plot_multi, plot_medium, plot_initial, plot_error, plot_circuit
from .solvers import Solver1DODE, Solver1DEXP, Solver1DLocal, Solver1DCloud

# -------- CONSTANTS --------
EXT = '.pkl'

# -------- CLASSES --------
class ForwardExperiment1D:
    """
    Class for running a 1D forward elastic wave simulation experiment
    through different implementations.
    """
    def __init__(self, experiment_id=None, verbose=6, data_folder='data'):
        self.solvers = []
        self.data = {}
        self.configs = {}
        self.base_data = pathlib.Path(data_folder)
        if experiment_id:
            self.timestamp = experiment_id
            self.base_data = pathlib.Path(self.base_data)/self.timestamp
            if not self.base_data.exists():
                raise FileNotFoundError(f'No experiment with time stamp {self.timestamp} found.')
            self.logger = Logger.setup_logger(self.base_data/'log.log', verbose)

        else:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            self.base_data = self.base_data/self.timestamp
            self.base_data.mkdir(parents=True, exist_ok=True)
            self.logger = Logger.setup_logger(self.base_data/'log.log', verbose)
            self.logger.info(f'Created experiment with time stamp: {self.timestamp}.\n')

    def add_solver(self, solver: str, dx: float, nx: int, dt: float,  nt: int, order: int,
                   bcs: dict, mu: np.ndarray, rho: np.ndarray,
                   u: np.ndarray, v: np.ndarray, backend: dict):
        """
        Add a 1D solver to the experiment.
        
        Args:
            solver (str): Solver type. One of 'ode', 'exp', 'local', 'cloud'.
            dx (float): Spatial step size.
            nx (int): Number of spatial grid points.
            dt (float): Temporal step size.
            nt (int): Number of temporal grid points.
            order (int): FD Order of the solver.
            bcs (dict): Boundary conditions.
            mu (np.ndarray): Medium elastic moduli.
            rho (np.ndarray): Medium densities.
            u (np.ndarray): Initial condition for positions.
            v (np.ndarray): Initial condition for velocities.
            backend (dict): Backend configuration.
        """
        self.logger.info(f'Adding solver {len(self.solvers)+1}: {solver}')

        # Check solver
        assert solver in ['ode', 'exp', 'local', 'cloud'], 'Solver not implemented.'

        # Define solver number
        idx = len(self.solvers)

        # Define kwargs
        kwargs = {
            'idx': idx,
            'solver': solver,
            'dx': dx,
            'nx': nx,
            'dt': dt,
            'nt': nt,
            'order': order,
            'bcs': bcs,
            'mu': mu,
            'rho': rho,
            'u': u,
            'v': v,
            'backend': backend
        }

        config = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        self.configs[idx] = config
        self.logger.debug(json.dumps(config, indent=4))

        # Add solver
        match solver:
            case 'ode':
                self.solvers.append(Solver1DODE(self.base_data, self.logger, **kwargs))
            case 'exp':
                self.solvers.append(Solver1DEXP(self.base_data, self.logger, **kwargs))
            case 'local':
                self.solvers.append(Solver1DLocal(self.base_data, self.logger, **kwargs))
            case 'cloud':
                self.solvers.append(Solver1DCloud(self.base_data, self.logger, **kwargs))
        self.logger.info(f'Solver {idx} added.\n')

    def run(self):
        """
        Run the experiment.
        
        Returns:
            dict: Simulation results.
        """
        # Save configs
        json.dump(self.configs, open(self.base_data/'configs.json', 'w', encoding='utf8'), indent=4)

        # Run solvers
        for idx, solver in enumerate(self.solvers):
            start_time = datetime.datetime.now()
            self.logger.info(f'Running solver {idx+1}: {solver.kwargs["solver"]}')
            self.data[idx] = solver.run()
            end_time = datetime.datetime.now()
            self.logger.info('Saving data.')
            pickle.dump(self.data[idx], open(self.base_data/f'data_{idx}{EXT}', 'wb'))
            self.logger.info(f'Solver {idx+1} completed in {end_time-start_time}.\n')
        return self.data

    def load(self):
        """
        Load the experiment.
        
        Returns:
            dict: Simulation results.
        """
        self.logger.info(f'Loading experiment with time stamp: {self.timestamp}.\n')

        # Load configs
        if not (self.base_data/'configs.json').exists():
            raise FileNotFoundError('No existing experiment found.')
        self.configs = json.load(open(self.base_data/'configs.json', 'r', encoding='utf8'))

        # Load data
        for idx in self.configs.keys():
            idx = int(idx)
            self.logger.info(f'Loading data for solver {idx+1}.')
            if not (self.base_data/f'data_{idx}{EXT}').exists():
                if self.configs[str(idx)]['solver'] == 'cloud':
                    self.logger.warning(f'No data for solver {idx+1} found. Loading from cloud.')
                    solver = Solver1DCloud(self.base_data,
                                                      self.logger, **self.configs[str(idx)])
                    self.data[idx] = solver.load()
                    self.logger.info(f'Data for solver {idx+1} loaded from cloud.')
                    pickle.dump(self.data[idx], open(self.base_data/f'data_{idx}{EXT}', 'wb'))
                else:
                    raise FileNotFoundError(f'No data for solver {idx+1} found.')
            else:
                self.data[idx] = pickle.load(open(self.base_data/f'data_{idx}{EXT}', 'rb'))

        self.logger.info('Data loaded.\n')
        return self.data

    def plot(self, mode, solvers=None, **kwargs):
        """
        Plot the experiment.
        
        Args:
            mode (str): Plotting mode. One of 'multi', 'medium', 'initial', 'error','circuit'.
            solvers (list): List of solvers to plot. Defaults to None.
            **kwargs: Keyword arguments for plotting functions.
            
        Returns:
            matplotlib.pyplot.figure: Figure handle.
        """
        match mode:
            case 'multi':
                return plot_multi([self.data[solver] for solver in solvers], **kwargs)
            case 'medium':
                assert len(solvers) == 1, 'Please provide exactly one solver for medium plotting.'
                plot_medium(self.configs[solvers[0]]['mu'],
                            self.configs[solvers[0]]['rho'], **kwargs)
            case 'initial':
                assert len(solvers) == 1, 'Please provide exactly one solver for initial plotting.'
                bcs = self.configs[solvers[0]]['bcs']
                plot_initial(self.configs[solvers[0]]['u'], self.configs[solvers[0]]['v'],
                             bcs, **kwargs)
            case 'error':
                assert len(solvers) == 2, 'Please provide exactly two solvers for error plotting.'
                return plot_error(self.data[solvers[0]], self.data[solvers[1]], **kwargs)
            case 'circuit':
                assert len(solvers) == 1, 'Please provide exactly one solver for circuit plotting.'
                return plot_circuit(self.solvers[solvers[0]], **kwargs)
            case _:
                raise ValueError('Invalid plotting mode.')
