#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for performing a real-valued quantum state tomography
on the measurement results of a quantum experiment.}

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Other modules
import numpy as np
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_experiments.library.tomography.fitters import (linear_inversion,
                                                           cvxpy_linear_lstsq, cvxpy_gaussian_lstsq)

# -------- CONSTANTS --------
FITTER_FUNCTIONS = {
    'linear': linear_inversion,
    'cvxpy_gaussian': cvxpy_gaussian_lstsq,
    'cvxpy_linear': cvxpy_linear_lstsq,
}

# -------- CLASSES --------
class TomographyReal:
    """
    Class that performs a state tomography on the measurement results of a quantum experiment.
    """

    def __init__(self, logger: object, fitter: str) -> None:
        self.logger = logger
        self.fitter = fitter

    def run_tomography(self, measurements: list, observables: list, times: list) -> np.ndarray:
        """
        Run a state tomography on the measurement results of a quantum experiment.
        
        Args:
            measurements (list): The measurement results of the experiment.
            observables (list): The observables that were measured.
            times (list): The times at which the tomography is performed.
            
        Returns:
            np.ndarray: The tomography result states.
        """

        quasi_dist = [dist for m in measurements for dist in m.quasi_dists]
        max_key = max(max(dist.keys()) for dist in quasi_dist)
        _ = [dist.setdefault(i, 0) for dist in quasi_dist for i in range(max_key + 1)]
        freq = np.array([dist[key] for dist in quasi_dist for key in range(max_key + 1)])
        freq = np.reshape(freq, (len(quasi_dist), max_key+1))
        shot_data = np.array([measurements[0].metadata[0]['shots']] * len(observables))
        measurement_data = np.array([[{'Z': 0, 'X': 1, 'Y': 2}[char] for char in sublist]
                                     for sublist in observables])
        preparation_data = np.full((len(shot_data), 0), None)

        fitter_kwargs = {
            "measurement_basis": PauliMeasurementBasis(),
            "measurement_qubits": tuple(range(len(measurement_data[0])))
        }

        def process_time_step(i):
            self.logger.info(f'Tomography step: {i+1} | {len(times)}.')
            freq_step = freq[i*len(observables):(i+1)*len(observables)]
            outcome_data = np.expand_dims(freq_step * shot_data[:, np.newaxis], axis=0)
            rho, metadata = FITTER_FUNCTIONS[self.fitter](
                outcome_data, shot_data, measurement_data, preparation_data, **fitter_kwargs)
            eigval, eigvec = np.linalg.eig(rho)
            self.logger.debug(f'Eigenvalues: {eigval}')
            self.logger.debug(f'Eigenvectors: {eigvec}')
            self.logger.debug(f'Metadata: {metadata}')
            state = eigvec[:, np.argmax(eigval)]
            return state

        states = np.zeros((len(times), 2**len(observables[0])), dtype=complex)
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(process_time_step, i): i for i in range(len(times))}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                states[index] = future.result()
        return states

# -------- FUNCTIONS --------
def parallel_transport(states: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """
    Parallel transport phase correction of states.
    
    Args:
        states (np.ndarray): The states to be corrected.
        initial_state (np.ndarray): The initial state.
        
    Returns:
        np.ndarray: The corrected states.
    """
    corrected_states = [initial_state]
    for state in states:
        ref_state = corrected_states[-1]
        phase = np.imag(np.log(state.dot(np.conj(ref_state))))
        corrected_states.append(np.exp(-1j * phase) * state)
    return np.array(corrected_states)
