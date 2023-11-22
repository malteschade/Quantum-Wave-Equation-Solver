from simulation.solvers import Solver1DODE, Solver1DLocal, Solver1DCloud

class ForwardExperiment1D:
    def __init__(self):
        self.solvers = []
        self.results = {}
    
    def add_solver(self, solver: str, dx: float, nx: int, dt: float,  nt: int, order: int, bcs: dict,
                   mu: list, rho: list,  u: list, v: list, backend: dict):
        
        # Checks
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
        
        # Add solver
        match solver:
            case 'ode':
                self.solvers.append(Solver1DODE(**kwargs))
            case 'local':
                self.solvers.append(Solver1DLocal(**kwargs))
            case 'cloud':
                self.solvers.append(Solver1DCloud(**kwargs))

            
    def run(self):
        for i, solver in enumerate(self.solvers):
            self.results[i] = solver.run()
        return self.results
            
    def save_experiment(self):
        pass
    
    def load_experiment(self):
        self.results = 'test'
        return self.results
    