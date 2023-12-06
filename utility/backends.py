#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for handling cloud and local backends.}

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
        self.options = self.init_options(kwargs['fake'], kwargs['seed'],
                                         kwargs['shots'], kwargs['optimization'],
                                         kwargs['resilience'], kwargs['local_transpilation'])

    def init_options(self, fake: str, seed: int, shots: int,
                     optimization: int, resilience: int, local_transpilation: bool) -> Options:
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
            skip_transpilation (bool): Skip transpilation in cloud service.

        Returns:
            Options: An object containing various settings for the backend, including simulation, 
                     transpilation, and execution options.
        """
        self.fake_backend = FAKE_PROVIDERS.get(fake, None)
        simulator_options = {
            "seed_simulator": seed,
            "coupling_map": self.fake_backend.coupling_map if self.fake_backend else None,
            "noise_model": NoiseModel.from_backend(self.fake_backend) if self.fake_backend else None
            }
        transpile_options = {"skip_transpilation": local_transpilation}
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
        self.logger.info('Connecting to QiskitRuntimeService.')
        backend_service = BackendService()
        self.service = backend_service.service
        self.backend = backend_service.backends.get(kwargs['backend'], None)
        self.logger.info('Connection successful.')

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
        self.logger.debug(f'Backend options: {self.options}')
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
        self.logger.debug(f'Backend options: {self.options}')
        sampler = AerSampler(
        backend_options={**self.options.simulator.__dict__,
                            **{'method': self.kwargs['method'],
                            'max_parallel_experiments': self.kwargs['max_parallel_experiments']}},
        transpile_options={"seed_transpiler": self.kwargs['seed']},
        run_options=self.options.execution.__dict__)
        return (sampler, None)
