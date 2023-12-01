#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
# Built-in modules
import warnings
from typing import List, Dict
from itertools import product

# Other modules
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import ProductFormula, SuzukiTrotter, LieTrotter, QDrift, MatrixExponential
from qiskit.compiler import transpile

# -------- CONSTANTS --------
SYNTHESIS: Dict[str, ProductFormula] = {
    'MatrixExponential': MatrixExponential(),
    'LieTrotter': LieTrotter(reps=3),
    'SuzukiTrotter': SuzukiTrotter(order=2, reps=2),
    'QDrift': QDrift(reps=4)
}

# -------- CLASSES --------
class CircuitGen1DA:
    """
    A class for generating quantum circuits for 1D elastic wave forward problems.

    This class is responsible for creating quantum circuits based on given Hamiltonians,
    initial states, and measurement bases. It supports various methods of Hamiltonian
    synthesis.

    Args:
        logger: A logging instance for recording the generation process.

    Attributes:
        logger (object): Logger for recording information.
        meas_circuits (dict): Predefined measurement circuits.
    """
    def __init__(self, logger: object, backend: object = None) -> None:
        """
        Initializes the CircuitGen1DA class with a logger.

        Args:
            logger: A logging instance.
        """
        self.logger = logger
        self.backend = backend
        self.meas_circuits = self.measurement_circuits()

    def tomography_circuits(self, initial_state: np.ndarray, hamiltonian: np.ndarray,
                            times: np.ndarray, synthesis: str = 'MatrixExponential',
                            batch_size: int = 100) -> List[List[QuantumCircuit]]:
        """
        Generates quantum circuits for tomography based on the given parameters.

        Args:
            initial_state (np.ndarray): The initial state vector for the quantum system.
            hamiltonian (np.ndarray): The Hamiltonian matrix representing the system dynamics.
            times (List[float]): A list of times at which to apply the evolution.
            synthesis (str): The method of Hamiltonian synthesis to use.
                             Default is 'MatrixExponential'.
            batch_size (int): The number of circuits per batch. Default is 100.

        Returns:
            List[List[QuantumCircuit]]: A list of lists containing quantum circuits in batches.
        """
        num_qubits = int(np.log2(hamiltonian.shape[0]))
        observables = list(product("ZX", repeat=num_qubits))
        synthesis = SYNTHESIS[synthesis]
        op = SparsePauliOp.from_operator(Operator(hamiltonian))

        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)

        self.logger.debug(initial_state)
        n = len(initial_state)
        if np.all(np.nonzero(initial_state)[0] == [n//4, n//4+1]):
            self.logger.info('Preparing efficient initial state (central spike).')
            qc.ry(-2*np.arcsin(initial_state[n//4+1]), 0)
            qc.x(num_qubits-2)
            qc.z(num_qubits-2)
        else:
            self.logger.info('Preparing arbitrary initial state.')
            qc.append(StatePreparation(initial_state), qr)
        qc.barrier()

        circuits = []
        for idx, time in enumerate(times):
            self.logger.info(f'Generating circuits for step: {idx+1} | {len(times)}.')
            evo = PauliEvolutionGate(op, time=time, synthesis=synthesis)
            qc_evolution = qc.copy()
            qc_evolution.append(evo.definition, qr)
            qc_evolution.barrier()

            for observable in observables:
                qc_measurement = qc_evolution.copy()
                for i, obs in enumerate(observable):
                    qc_measurement.append(self.meas_circuits[obs], [i])
                qc_measurement.barrier()
                qc_measurement.measure(qr, cr)

                if self.backend:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        qc_measurement = transpile(
                            qc_measurement,
                            backend=self.backend,
                            optimization_level=3,
                            seed_transpiler=0
                        )

                        self.logger.debug(
                            f'Circuit depth after transp.: {qc_measurement.depth()}'
                        )

                    if caught_warnings:
                        for _ in caught_warnings:
                            #self.logger.debug(warning.message)
                            pass
                circuits.append(qc_measurement)

        circuit_groups = [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]
        n_circuits = len(times)*len(observables)
        self.logger.info(f'Circuits generated: {n_circuits} in {len(circuit_groups)} groups.')
        self.logger.info(f'Circuit depth on backend: {circuits[0].depth()}')
        return circuit_groups

    def measurement_circuits(self) -> Dict[str, QuantumCircuit]:
        """
        Creates basic measurement circuits for Pauli measurements.

        Returns:
            Dict[str, QuantumCircuit]: A dictionary mapping 'Z', 'X', and 'Y' to their respective
                                       measurement circuits.
        """
        meas_z = QuantumCircuit(1, name="PauliMeasZ")
        meas_x = QuantumCircuit(1, name="PauliMeasX")
        meas_y = QuantumCircuit(1, name="PauliMeasY")
        meas_x.h(0)
        meas_y.sdg(0)
        meas_y.h(0)
        return {"Z" : meas_z, "X" : meas_x, "Y" : meas_y}
