#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
# Built-in modules
import datetime
import pathlib
import json

# Other modules
import numpy as np

# Own modules
from simulation.solvers import Solver1DODE, Solver1DLocal, Solver1DCloud
from config.logger import Logger

# -------- CLASSES --------
class ForwardExperiment1D:
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
                   bcs: dict, mu: list, rho: list,  u: list, v: list, backend: dict):
        self.logger.info(f'Adding solver {len(self.solvers)+1}: {solver}')

        # Check solver
        assert solver in ['ode', 'local', 'cloud'], 'Solver not implemented.'

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
            case 'local':
                self.solvers.append(Solver1DLocal(self.base_data, self.logger, **kwargs))
            case 'cloud':
                self.solvers.append(Solver1DCloud(self.base_data, self.logger, **kwargs))
        self.logger.info(f'Solver {idx} added.\n')

    def run(self):
        # Save configs
        json.dump(self.configs, open(self.base_data/'configs.json', 'w', encoding='utf8'), indent=4)

        # Run solvers
        for idx, solver in enumerate(self.solvers):
            start_time = datetime.datetime.now()
            self.logger.info(f'Running solver {idx+1}: {solver.kwargs["solver"]}')
            self.data[idx] = solver.run()
            end_time = datetime.datetime.now()
            self.logger.info('Saving data.')
            json.dump(self.data[idx], open(self.base_data/f'data_{idx}.json',
                                            'w', encoding='utf8'))
            self.logger.info(f'Solver {idx+1} completed in {end_time-start_time}.\n')
        return self.data

    def load(self):
        self.logger.info(f'Loading experiment with time stamp: {self.timestamp}.\n')

        # Load configs
        if not (self.base_data/'configs.json').exists():
            raise FileNotFoundError('No existing experiment found.')
        self.configs = json.load(open(self.base_data/'configs.json', 'r', encoding='utf8'))

        # Load data
        for idx in self.configs.keys():
            idx = int(idx)
            self.logger.info(f'Loading data for solver {idx+1}.')
            if not (self.base_data/f'data_{idx}.json').exists():
                if self.configs[str(idx)]['solver'] == 'cloud':
                    self.logger.warning(f'No data for solver {idx+1} found. Loading from cloud.')
                    solver = Solver1DCloud(self.base_data,
                                                      self.logger, **self.configs[str(idx)])
                    self.data[idx] = solver.load()
                    self.logger.info(f'Data for solver {idx+1} loaded from cloud.')
                    json.dump(self.data[idx], open(self.base_data/f'data_{idx}.json',
                                                    'w', encoding='utf8'))
                else:
                    raise FileNotFoundError(f'No data for solver {idx+1} found.')
            else:
                self.data[idx] = json.load(open(self.base_data/f'data_{idx}.json',
                                                'r', encoding='utf8'))
        self.logger.info('Data loaded.\n')
        return self.data
