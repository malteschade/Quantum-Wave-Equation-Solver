import datetime
import pathlib
import json

from simulation.solvers import Solver1DODE, Solver1DLocal, Solver1DCloud
from config.logger import Logger, handle_ndarray

class ForwardExperiment1D:
    def __init__(self, verbose=6, data_folder='data'):
        self.solvers = []
        self.results = {}
        self.base_data = pathlib.Path(data_folder)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.base_data = self.base_data/self.timestamp
        self.base_data.mkdir(parents=True, exist_ok=True)
        self.logger = Logger.setup_logger(self.base_data/'log.log', verbose)
        self.logger.info(f'Created experiment with time stamp: {self.timestamp}.\n')
    
    def add_solver(self, solver: str, dx: float, nx: int, dt: float,  nt: int, order: int, bcs: dict,
                   mu: list, rho: list,  u: list, v: list, backend: dict):
        self.logger.info(f'Adding solver {len(self.solvers)+1}: {solver}')
        
        # Check solver
        assert solver in ['ode', 'local', 'cloud'], 'Solver not implemented.'
        
        # Define kwargs
        kwargs = {
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
        self.logger.debug(json.dumps({k: handle_ndarray(v) for k, v in kwargs.items()}, indent=4))
        
        # Add solver
        match solver:
            case 'ode':
                self.solvers.append(Solver1DODE(self.logger, **kwargs))
            case 'local':
                self.solvers.append(Solver1DLocal(self.logger, **kwargs))
            case 'cloud':
                self.solvers.append(Solver1DCloud(self.logger, **kwargs))
        self.logger.info(f'Solver {len(self.solvers)} added.\n')

            
    def run(self):
        for i, solver in enumerate(self.solvers):
            start_time = datetime.datetime.now()
            self.logger.info(f'Running solver {i+1}: {solver.kwargs["solver"]}')
            self.results[i] = solver.run()
            end_time = datetime.datetime.now()
            self.logger.info(f'Solver {i+1} completed in {end_time-start_time}.\n')
        return self.results
            
    def save_experiment(self):
        pass
    
    def load_experiment(self):
        pass
    