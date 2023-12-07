#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for implementing a forward simulation experiment
with different wave equation solvers. Handling of data saving
and data loading.}

{
    MIT License

    Copyright (c) [2023] [Malte Leander Schade]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
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
from utility.plotting import plot_multi, plot_medium, plot_initial, plot_error
from .solvers import Solver1DODE, Solver1DEXP, Solver1DLocal, Solver1DCloud

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
            pickle.dump(self.data[idx], open(self.base_data/f'data_{idx}.pkl', 'wb'))
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
            if not (self.base_data/f'data_{idx}.pkl').exists():
                if self.configs[str(idx)]['solver'] == 'cloud':
                    self.logger.warning(f'No data for solver {idx+1} found. Loading from cloud.')
                    solver = Solver1DCloud(self.base_data,
                                                      self.logger, **self.configs[str(idx)])
                    self.data[idx] = solver.load()
                    self.logger.info(f'Data for solver {idx+1} loaded from cloud.')
                    pickle.dump(self.data[idx], open(self.base_data/f'data_{idx}.pkl', 'wb'))
                else:
                    raise FileNotFoundError(f'No data for solver {idx+1} found.')
            else:
                self.data[idx] = pickle.load(open(self.base_data/f'data_{idx}.pkl', 'rb'))
        self.logger.info('Data loaded.\n')
        return self.data

    def plot(self, mode, solvers=None, **kwargs):
        """
        Plot the experiment.
        
        Args:
            mode (str): Plotting mode. One of 'multi', 'medium', 'initial', 'error'.
            solvers (list): List of solvers to plot. Defaults to None.
            **kwargs: Keyword arguments for plotting functions.
            
        Returns:
            matplotlib.pyplot.figure: Figure handle.
        """
        match mode:
            case 'multi':
                assert len(solvers) == 3, 'Please provide exactly three solvers for multi plotting.'
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
