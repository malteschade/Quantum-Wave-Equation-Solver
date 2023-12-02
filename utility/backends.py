#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Dict, Tuple

# Other modules
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import (FakeBackendV2, FakeSherbrooke, FakeGuadalupeV2)

# -------- CONSTANTS --------
FAKE_PROVIDERS: Dict[str, FakeBackendV2] = {
    'guadalupe': FakeGuadalupeV2(),
    'brisbane': FakeSherbrooke(),
    'kyoto': FakeSherbrooke(),
    'sherbrooke': FakeSherbrooke()
    }

# -------- CLASSES --------
class BackendService:
    """
    A service for managing backend instances.

    This class uses the singleton pattern to ensure that only one instance
    of the service and its associated backends are created.

    Attributes:
        _instance (Optional[BackendService]): A static instance of BackendService.
        service (QiskitRuntimeService): An instance of QiskitRuntimeService.
        backends (Dict[str, object]): A dictionary of backend instances.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BackendService, cls).__new__(cls, *args, **kwargs)
            cls.service = QiskitRuntimeService()
            cls.backends = {backend.name: backend for backend in cls.service.backends()}
        return cls._instance

class BaseBackend:
    """
    A base class for quantum computing backends.

    This class initializes the backend options based on the provided arguments.

    Args:
        **kwargs: Arbitrary keyword arguments for backend configuration.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.options = self.init_options(kwargs['backend'], kwargs['fake'], kwargs['seed'],
                                         kwargs['shots'], kwargs['optimization'],
                                         kwargs['resilience'])

    def init_options(self, backend: str, fake: str, seed: int, shots: int,
                     optimization: int, resilience: int) -> Options:
        """
        Initializes and returns the options for the quantum computing backend.

        This method sets up the backend (real or simulated), simulator options, 
        transpilation options, and execution options based on the provided arguments.

        Args:
            backend (str): The name of the real backend to use.
            fake (str): The name of the fake backend to use.
            seed (int): Seed for the simulator. This controls the randomness of the simulation.
            shots (int): The number of shots (repetitions) for each circuit execution.
            optimization (int): The level of optimization to use for the transpiler.
            resilience (int): The level of resilience to noise to use for the transpiler.

        Returns:
            Options: An object containing various settings for the backend, including simulation, 
                     transpilation, and execution options.
        """

        self.service = BackendService().service if backend else None
        self.backend = BackendService().backends.get(backend, None) if backend else None
        self.fake_backend = FAKE_PROVIDERS.get(fake, None)
        simulator_options = {
            "seed_simulator": seed,
            "coupling_map": self.fake_backend.coupling_map if self.fake_backend else None,
            "noise_model": NoiseModel.from_backend(self.fake_backend) if self.fake_backend else None
            }
        transpile_options = {"skip_transpilation": False} # Important
        run_options = {"shots": shots}

        return Options(
            optimization_level=optimization,
            resilience_level=resilience,
            transpilation=transpile_options,
            execution=run_options,
            simulator=simulator_options
            )

class CloudBackend(BaseBackend):
    """
    A backend class for cloud-based quantum computing services.

    This class initializes a cloud-based backend and provides a method to get a sampler
    for quantum computations.

    Args:
        logger: A logging instance.
        **kwargs: Arbitrary keyword arguments for backend configuration.
    """

    def __init__(self, logger: object, **kwargs) -> None:
        self.logger = logger
        self.logger.info('Setting backend options.')
        super().__init__(**kwargs)

    def get_sampler(self) -> Tuple[Sampler, Session]:
        """
        Initializes and returns a Sampler instance for cloud-based quantum computations.

        Raises:
            Exception: If the runtime service initialization fails.

        Returns:
            tuple: A tuple containing the Sampler instance and its associated Session.
        """

        self.logger.info('Initializing Runtime Service.')
        try:
            session = Session(service=self.service, backend=self.backend)
        except Exception as e:
            self.logger.error(f'Failed to initialize Runtime Service: {e}')
            raise e
        self.logger.info('Initializing sampler backend.')
        self.logger.debug(self.options)
        sampler = Sampler(session=session, options=self.options)
        return (sampler, session)

class LocalBackend(BaseBackend):
    """
    A backend class for local quantum computing simulations.

    This class initializes a local backend and provides a method to get a sampler
    for quantum computations.

    Args:
        logger: A logging instance.
        **kwargs: Arbitrary keyword arguments for backend configuration.
    """
    def __init__(self, logger: object, **kwargs) -> None:
        self.logger = logger
        self.logger.info('Setting backend options.')
        super().__init__(**kwargs)

    def get_sampler(self) -> Tuple[AerSampler, None]:
        """
        Initializes and returns an AerSampler instance for local quantum computations.

        Returns:
            tuple: A tuple containing the AerSampler instance and None (since no session is needed).
        """

        self.logger.info('Initializing sampler backend.')
        self.logger.debug(self.options)
        sampler = AerSampler(
        backend_options={**self.options.simulator.__dict__,
                            **{'method': self.kwargs['method'],
                            'max_parallel_experiments': self.kwargs['max_parallel_experiments']}},
        transpile_options={"seed_transpiler": self.kwargs['seed']},
        run_options=self.options.execution.__dict__)
        return (sampler, None)
