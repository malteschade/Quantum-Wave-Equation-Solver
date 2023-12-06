#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}

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
# Other modules
import numpy as np
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_experiments.library.tomography.fitters import (linear_inversion,
                                                           scipy_linear_lstsq, scipy_gaussian_lstsq,
                                                           cvxpy_linear_lstsq, cvxpy_gaussian_lstsq)

# -------- CONSTANTS --------
FITTER_FUNCTIONS = {
    'linear': linear_inversion,
    'cvxpy_gaussian': cvxpy_gaussian_lstsq,
    'cvxpy_linear': cvxpy_linear_lstsq,
    'scipy_gaussian': scipy_gaussian_lstsq,
    'scipy_linear': scipy_linear_lstsq,
}

# -------- CLASSES --------
class TomographyReal:
    """
    Class that performs a state tomography on the measurement results of a quantum experiment.
    """

    def __init__(self, logger: object, fitter: str) -> None:
        self.logger = logger
        self.fitter = fitter

    def run_tomography(self, result_groups: list, observables: list, times: list) -> np.ndarray:
        """
        Run a state tomography on the measurement results of a quantum experiment.
        
        Args:
            result_groups (list): The measurement results of the experiment.
            observables (list): The observables that were measured.
            times (list): The times at which the tomography is performed.
            
        Returns:
            np.ndarray: The tomography result states.
        """

        measurements = [result for result_group in result_groups
                        for result in result_group.decompose()]
        quasi_dist = [m.quasi_dists[0] for m in measurements]
        max_key = max(max(dist.keys()) for dist in quasi_dist)
        _ = [dist.setdefault(i, 0) for dist in quasi_dist for i in range(max_key + 1)]
        freq = np.array([dist[key] for dist in quasi_dist for key in range(max_key + 1)])
        freq = np.reshape(freq, (len(quasi_dist), max_key+1))

        shot_data = np.array([meta['shots'] for meta in measurements[0].metadata]*len(observables))
        measurement_data = np.array([[{'Z': 0, 'X': 1, 'Y': 2}[char] for char in sublist]
                                     for sublist in observables])
        preparation_data = np.full((len(shot_data), 0), None)

        fitter_kwargs = {
            "measurement_basis": PauliMeasurementBasis(),
            "measurement_qubits": tuple(range(len(measurement_data[0])))
        }

        states = []
        for i in range(len(times)):
            self.logger.info(f'Tomography step: {i+1} | {len(times)}.')
            freq_step = freq[i*len(observables):(i+1)*len(observables)]
            outcome_data = np.expand_dims(freq_step * shot_data[:, np.newaxis], axis=0)
            rho, metadata = FITTER_FUNCTIONS[self.fitter](
                outcome_data, shot_data,measurement_data, preparation_data, **fitter_kwargs)
            eigval, eigvec = np.linalg.eig(rho)
            self.logger.debug(f'Eigenvalues: {eigval}')
            self.logger.debug(f'Eigenvectors: {eigvec}')
            self.logger.debug(f'Metadata: {metadata}')
            state = eigvec[:, np.argmax(eigval)]
            states.append(state)
        return np.array(states)

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
