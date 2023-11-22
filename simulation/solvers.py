from time import sleep

import numpy as np

from scipy.integrate import odeint

from utility.transform import FDTransform1DA
from utility.processing import MediumProcessor, StateProcessor
from utility.backends import CloudBackend, LocalBackend
from utility.circuits import CircuitGen1DA
from utility.tomography import TomographyReal, parallel_transport

class Solver1D:
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.kwargs = kwargs
        self.data = {'config': self.kwargs}
        
        # Check parameters
        self.check_kwargs()
        self.logger.info(f'Parameters checked for validity.')
        
        # Set time steps
        self.set_times()
        self.logger.info(f'Solving for {self.kwargs["nt"]} time steps.')
        self.logger.debug(f'Times: {self.data["times"]}')
        
        # Set transform
        self.logger.info(f'Calculating Transformation and Hamiltonian.')
        self.set_transform()
        self.logger.info(f'Calculation completed.')
        
        # Set medium
        self.set_medium()
        self.logger.info(f'Medium initialized.')
        
    def check_kwargs(self):
        assert self.kwargs['nx'] > 0, 'nx must be greater than zero'
        assert np.log2(self.kwargs['nx']+1) % 1 == 0, 'nx must be a power of two minus one'
        assert self.kwargs['nx'] == len(self.kwargs['mu'])-1, 'length of mu must be one more than nx'
        assert self.kwargs['nx'] == len(self.kwargs['rho']), 'length of rho must be equal to nx'
        assert np.all(self.kwargs['mu'] > 0), 'mu must be positive'
        assert np.all(self.kwargs['rho'] > 0), 'rho must be positive'
        assert self.kwargs['nx'] == len(self.kwargs['u']), 'length of u must be equal to nx'
        assert self.kwargs['nx'] == len(self.kwargs['v']), 'length of v must be equal to nx'
        assert self.kwargs['nt'] > 0, 'nt must be greater than zero'
        assert self.kwargs['dt'] > 0, 'dt must be greater than zero'
        assert self.kwargs['dx'] > 0, 'dx must be greater than zero'
        assert self.kwargs['order'] in [1,2,3,4], "Order must be in [1,2,3,4]"
        assert self.kwargs['bcs']['left'] in ['DBC', 'NBC'], "Left boundary condition must be DBC or NBC"
        assert self.kwargs['bcs']['right'] in ['DBC', 'NBC'], "Right boundary condition must be DBC or NBC"
        
    def set_times(self):
        self.times = np.arange(self.kwargs['nt'])*self.kwargs['dt']
        self.data['times'] = self.times
        
    def set_transform(self):
        # Initialize transform
        self.tf = FDTransform1DA(
            self.kwargs['mu'],
            self.kwargs['rho'],
            self.kwargs['dx'],
            self.kwargs['nx'],
            self.kwargs['order'],
            self.kwargs['bcs']
            )
        self.data['transform'] = self.tf.get_dict()
        
    def set_medium(self):
        # Initialize medium processor
        self.md = MediumProcessor(len(self.kwargs['mu']), len(self.kwargs['rho']))
        
        # Set medium parameters
        self.md.set_mu(self.kwargs['mu'])
        self.md.set_rho(self.kwargs['rho'])
        
        # Get medium parameters
        self.medium = self.md.get_medium()
        self.data['medium'] = self.md.get_dict()

class Solver1DODE(Solver1D):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=0)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.sqrt_m)
        self.logger.info(f'Initial state forward-transformed.')
        
    def run(self):
        self.logger.info(f'Solving ODE.')
        self.st.states = odeint(lambda y, t: self.tf.q @ y, self.st.get_state(0), self.times)
        self.logger.info(f'ODE solved.')
        [self.st.inverse_state(i, self.tf.inv_sqrt_m) for i in range(len(self.times))]
        self.logger.info(f'States inverse-transformed.')
        
        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DLocal(Solver1D):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.logger.info(f'Initial state transformed.')
        
    def run(self):
        self.logger.info(f'Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger)
        circuit_groups = circuit_gen.tomography_circuits(self.st.get_state(0), self.tf.h, self.times[1:])
        
        self.logger.info(f'Initializing backend.')
        backend = LocalBackend()
        sampler = backend.sampler
        self.logger.info(f'Backend initialized.')
        
        self.logger.info(f'Submitting jobs to backend.')
        jobs = [sampler.run(circuits) for circuits in circuit_groups]
        self.logger.info(f'Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info(f'Jobs completed.')
        
        self.logger.info(f'Running tomography.')
        tomo = TomographyReal(self.logger, fitter='cvxpy_gaussian')
        states_raw = tomo.run_tomography(result_groups, circuit_gen.observables, self.times[1:])
        self.logger.info(f'Tomography completed.')
        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info(f'State polarization corrected.')
        [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t) for i in range(1, len(self.times))]
        self.logger.info(f'States inverse-transformed.')
        
        self.data['field'] = self.st.get_dict()
        return self.data
    
class Solver1DCloud(Solver1D):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.logger
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.logger.info(f'Initial state transformed.')
        
    def run(self):
        self.logger.info(f'Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger)
        circuit_groups = circuit_gen.tomography_circuits(self.st.get_state(0), self.tf.h, self.times[1:])
        
        self.logger.info(f'Initializing backend.')
        backend = CloudBackend()
        sampler = backend.sampler
        self.logger.info(f'Backend initialized.')
        
        self.logger.info(f'Submitting {len(circuit_groups)} jobs to backend.')
        jobs, job_ids = [], []
        for i, circuits in enumerate(circuit_groups):
            job_transmitted = False
            while not job_transmitted:
                try:
                    job = sampler.run(circuits)
                    job_transmitted = True
                except Exception as e:
                    self.logger.warning(f'Job transmission {i} failed with error {e}. Retrying in 5 seconds.')
                    sleep(5)
            self.logger.info(f'Job {i} submitted with job_id {job.job_id()}.')
            jobs.append(job)
            job_ids.append(job.job_id())
        self.logger.info(f'Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info(f'Jobs completed.')

        self.logger.info(f'Running tomography.')
        tomo = TomographyReal(self.logger, fitter='cvxpy_gaussian')
        states_raw = tomo.run_tomography(result_groups, circuit_gen.observables, self.times[1:])
        self.logger.info(f'Tomography completed.')
        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info(f'State polarization corrected.')
        [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t) for i in range(1, len(self.times))]
        self.logger.info(f'States inverse-transformed.')
        
        self.data['field'] = self.st.get_dict()
        return self.data

def _wait_for_completion(jobs, logger):
    all_completed = False
    while not all_completed:
        sleep(10)
        completed = [job.status().name == "DONE" for job in jobs]
        logger.info(f"Jobs completed: {sum(completed)} | {len(jobs)}")
        all_completed = all(completed)
        