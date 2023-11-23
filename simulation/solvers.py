#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Dict, List, Any
from time import sleep

# Other modules
import numpy as np
from scipy.integrate import odeint

# Own modules
from utility.transform import FDTransform1DA
from utility.processing import MediumProcessor, StateProcessor
from utility.backends import CloudBackend, LocalBackend
from utility.circuits import CircuitGen1DA
from utility.tomography import TomographyReal, parallel_transport

# -------- CLASSES --------
class Solver1D:
    """
    A class for solving 1D elastic wave forward problems using quantum computing methods.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.

    Attributes:
        logger (object): Logger for logging information.
        kwargs (dict): Dictionary of keyword arguments.
        data (dict): Dictionary to store various configurations and results.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        self.logger = logger
        self.kwargs = kwargs
        self.data = {'config': self.kwargs}

        # Check parameters
        self.check_kwargs()
        self.logger.info('Parameters checked for validity.')

        # Set time steps
        self.times = self.get_times(kwargs['nt'], kwargs['dt'])
        self.data['times'] = self.times
        self.logger.info(f'Solving for {self.kwargs["nt"]} time steps.')
        self.logger.debug(f'Times: {self.data["times"]}')

        # Set transform
        self.logger.info('Calculating Transformation and Hamiltonian.')
        self.tf = self.get_transform(kwargs['mu'], kwargs['rho'], kwargs['dx'],
                                        kwargs['nx'], kwargs['order'], kwargs['bcs'])
        self.data['transform'] = self.tf.get_dict()
        self.logger.info('Calculation completed.')

        # Set medium
        self.md = self.get_medium_processor(kwargs['mu'], kwargs['rho'])
        self.medium = self.md.get_medium()
        self.data['medium'] = self.md.get_dict()
        self.logger.info('Medium initialized.')

    def check_kwargs(self):
        """
        Validates the keyword arguments provided to the solver.

        Raises:
            AssertionError: If any of the conditions for the arguments are not met.
        """

        assert self.kwargs['nx'] > 0, 'nx must be greater than zero'
        assert np.log2(self.kwargs['nx']+1) % 1 == 0, 'nx must be a power of two minus one'
        assert self.kwargs['nx'] == len(self.kwargs['mu'])-1,'length of mu \
            must be one more than nx'
        assert self.kwargs['nx'] == len(self.kwargs['rho']), 'length of rho must be equal to nx'
        assert np.all(self.kwargs['mu'] > 0), 'mu must be positive'
        assert np.all(self.kwargs['rho'] > 0), 'rho must be positive'
        assert self.kwargs['nx'] == len(self.kwargs['u']), 'length of u must be equal to nx'
        assert self.kwargs['nx'] == len(self.kwargs['v']), 'length of v must be equal to nx'
        assert self.kwargs['nt'] > 0, 'nt must be greater than zero'
        assert self.kwargs['dt'] > 0, 'dt must be greater than zero'
        assert self.kwargs['dx'] > 0, 'dx must be greater than zero'
        assert self.kwargs['order'] in [1,2,3,4], "Order must be in [1,2,3,4]"
        assert self.kwargs['bcs']['left'] in ['DBC', 'NBC'], "Left boundary condition \
            must be DBC or NBC"
        assert self.kwargs['bcs']['right'] in ['DBC', 'NBC'], "Right boundary condition \
            must be DBC or NBC"

    def get_times(self, nt, dt) -> np.ndarray:
        """
        Sets up the time steps for the solver based on the 'nt' and 'dt' arguments.
        
        Args:
            nt (int): The number of time steps.
            dt (float): The time step size.
            
        Returns:
            np.ndarray: An array of time steps.
        """

        return np.arange(nt)*dt

    def get_transform(self, mu: np.ndarray, rho: np.ndarray, dx: float, nx: int,
                      order: int, bcs: Dict[str, str]) -> FDTransform1DA:
        """
        Initializes and returns a finite difference transformation object for 1D analysis.

        This method creates an instance of the FDTransform1DA class, which is used for
        transforming the state space based on the provided medium parameters.

        Args:
            mu (np.ndarray): An array of µ values representing the medium shear modulus.
            rho (np.ndarray): An array of ρ values representing the medium density.
            dx (float): The spatial step size.
            nx (int): The number of spatial steps.
            order (int): The order of the finite difference scheme.
            bcs (Dict[str, str]): A dictionary specifying the boundary conditions
                                  with keys 'left' and 'right'.

        Returns:
            FDTransform1DA: An instance of FDTransform1DA initialized with the given parameters.
        """

        # Initialize transform
        return FDTransform1DA(mu, rho, dx, nx, order, bcs)

    def get_medium_processor(self, mu: np.ndarray, rho: np.ndarray) -> MediumProcessor:
        """
        Initializes and configures a medium processor for the simulation.

        This method creates an instance of MediumProcessor, setting it up with the 
        specified µ (mu) and ρ (rho) values representing the medium's properties. 

        Args:
            mu (np.ndarray): An array of µ values, representing the medium shear modulus.
            rho (np.ndarray): An array of ρ values, representing the medium density.

        Returns:
            MediumProcessor: An initialized and configured medium processor object.
        """

        # Initialize medium processor
        md =  MediumProcessor(len(mu), len(rho))

        # Set medium parameters
        md.set_mu(mu)
        md.set_rho(rho)

        return md

class Solver1DODE(Solver1D):
    """
    A subclass of Solver1D for solving with a classical\
        Ordinary Differential Equations (ODEs) solver.

    Inherits from Solver1D and adds specific methods for handling ODEs.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        super().__init__(logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=0)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.sqrt_m)
        self.logger.info('Initial state forward-transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the ODE solver and processes the results.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """
        self.logger.info('Solving ODE.')
        self.st.states = odeint(lambda y, t: self.tf.q @ y, self.st.get_state(0), self.times)
        self.logger.info('ODE solved.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m)
         for i in range(len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DLocal(Solver1D):
    """
    A subclass of Solver1D for local quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling local simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        super().__init__(logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the local solver, including quantum circuit generation, execution, and tomography.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger)
        circuit_groups = circuit_gen.tomography_circuits(self.st.get_state(0),
                                                         self.tf.h, self.times[1:])

        self.logger.info('Initializing backend.')
        backend = LocalBackend(self.logger, backend=None, fake=None, method='statevector',
                               max_parallel_experiments=0, seed=0, shots=10000,
                               optimization=3, resilience=1)
        sampler, _ = backend.get_sampler()
        self.logger.info('Backend initialized.')

        self.logger.info('Submitting jobs to backend.')
        jobs = [sampler.run(circuits) for circuits in circuit_groups]
        self.logger.info('Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, fitter='cvxpy_gaussian')
        states_raw = tomo.run_tomography(result_groups, circuit_gen.observables, self.times[1:])
        self.logger.info('Tomography completed.')
        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

class Solver1DCloud(Solver1D):
    """
    A subclass of Solver1D for cloud-based quantum computing simulations.

    Inherits from Solver1D and adds specific methods for handling cloud-based simulations.

    Args:
        logger (object): A logging instance to record the process and errors.
        **kwargs: Arbitrary keyword arguments for configuration.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        super().__init__(logger, **kwargs)
        self.st = StateProcessor(self.kwargs['nx'], self.kwargs['nt'], shift=1)
        self.st.set_u(self.kwargs['u'], 0)
        self.st.set_v(self.kwargs['v'], 0)
        self.st.forward_state(0, self.tf.t @ self.tf.sqrt_m)
        self.logger.info('Initial state transformed.')

    def run(self) -> Dict[str, Any]:
        """
        Runs the cloud-based solver, including quantum circuit generation,
        execution, and tomography.

        Returns:
            Dict[str, Any]: A dictionary containing the field data and other results.
        """

        self.logger.info('Generating circuits.')
        circuit_gen = CircuitGen1DA(self.logger)
        circuit_groups = circuit_gen.tomography_circuits(self.st.get_state(0),
                                                         self.tf.h, self.times[1:])

        self.logger.info('Initializing backend.')
        backend = CloudBackend(self.logger, backend='ibmq_qasm_simulator', fake=None,
                               seed=0, shots=10000, optimization=3, resilience=1)
        sampler, _ = backend.get_sampler()
        self.logger.info('Backend initialized.')

        self.logger.info(f'Submitting {len(circuit_groups)} jobs to backend.')
        jobs, job_ids = [], []
        for i, circuits in enumerate(circuit_groups):
            job_transmitted = False
            while not job_transmitted:
                try:
                    job = sampler.run(circuits)
                    job_transmitted = True
                except ConnectionError as e:
                    self.logger.warning(f'Job transmission {i} failed with error {e}.\
                                        Retrying in 5 seconds.')
                    sleep(5)
            self.logger.info(f'Job {i} submitted with job_id {job.job_id()}.')
            jobs.append(job)
            job_ids.append(job.job_id())
        self.logger.info('Jobs submitted.')
        _wait_for_completion(jobs, self.logger)
        result_groups = [job.result() for job in jobs]
        self.logger.info('Jobs completed.')

        self.logger.info('Running tomography.')
        tomo = TomographyReal(self.logger, fitter='cvxpy_gaussian')
        states_raw = tomo.run_tomography(result_groups, circuit_gen.observables, self.times[1:])
        self.logger.info('Tomography completed.')
        self.st.states = np.real(parallel_transport(states_raw, self.st.get_state(0)))
        self.logger.info('State polarization corrected.')
        _ = [self.st.inverse_state(i, self.tf.inv_sqrt_m @ self.tf.inv_t)
         for i in range(1, len(self.times))]
        self.logger.info('States inverse-transformed.')

        self.data['field'] = self.st.get_dict()
        return self.data

# -------- FUNCTIONS --------
def _wait_for_completion(jobs: List[object], logger: object) -> None:
    """
    Waits for a list of jobs to complete.

    Args:
        jobs (List[object]): A list of jobs to wait for.
        logger (object): A logger to record the status of the jobs.
    """

    all_completed = False
    while not all_completed:
        sleep(10)
        completed = [job.status().name == "DONE" for job in jobs]
        logger.info(f"Jobs completed: {sum(completed)} | {len(jobs)}")
        all_completed = all(completed)
