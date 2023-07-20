#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules
from functools import reduce
from itertools import product

# other modules
import numpy as np


__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'


# Define the Pauli matrices
bases = {'X': np.array([[0, 1], [1, 0]]),
         'Y': np.array([[0, -1j], [1j, 0]]),
         'Z': np.array([[1, 0], [0, -1]]),
         'I': np.array([[1, 0], [0, 1]])}


# Least squares state tomography
class LSStateTomo:
    def __init__(self, obs):
        self.povm = self.generate_povm_elements(obs)
        self.pred = self.build_state_predictor(self.povm)
        self.outputdim = self.pred.shape[0]
        self.inputdim = self.pred.shape[1]
        self.realpred = np.concatenate([np.real(self.pred), np.imag(self.pred)], axis=1)

    def generate_povm_elements(self, obs):
        tensor_product = lambda matrices: reduce(np.kron, matrices)
        
        povm = []
        for ob in obs:
            povm_basis = [[(bases['I'] + bases[b]) / 2, (bases['I'] - bases[b]) / 2] for b in ob]
            for combination in product(*povm_basis):
                povm.append(tensor_product(list(combination)))
        return povm
    
    def build_state_predictor(self, obs):
        return np.concatenate([np.reshape(o, -1)[np.newaxis, :] for o in obs], axis=0)
    
    def fit(self, means):
        d = int(np.sqrt(self.inputdim))
        reg, _, _, _ = np.linalg.lstsq(self.pred, means, rcond=None)
        rho = np.reshape(reg, (d, d))
        mse = np.linalg.norm(self.pred @ reg - means, 2) / len(means)
        return rho, mse


# Run tomography
def run_tomography(measurements, observables, state0, psi0, norm0, INV_T):
    # Define reference states and output array
    states = np.zeros((len(measurements), len(state0)))
    states[0], ref_state = state0, state0
    
    # Obtain Density matrices
    for i, meas in enumerate(measurements):
        # Pad non-measured entries with zeros
        max_key = max(max(d.keys()) for d in meas.quasi_dists)
        [m.setdefault(i, 0) for m in meas.quasi_dists for i in range(max_key + 1)]
        
        # Read measurement frequencies
        freq = [m[key] for m in meas.quasi_dists for key in range(max_key + 1)]
        
        # Run tomography
        tomo = LSStateTomo(observables)
        rho, mse = tomo.fit(freq)
        
        # Minimize phase difference with parallel transport (rho)
        values, vectors = np.linalg.eig(rho)
        psi_norm_2 = vectors[:, np.argmax(values)]
        
        # Minimize phase difference with parallel transport (psi)
        phase = np.imag(np.log(psi_norm_2.dot(np.conj(psi0))))
        psi_2 = np.exp(-1j * phase) * psi_norm_2 * norm0
        
        # Append state to states
        state = np.real(INV_T @ psi_2)

        # Minimize phase difference with parallel transport (state)
        phase = np.imag(np.log(state.dot(np.conj(ref_state))))
        ref_state = np.exp(-1j * phase) * state
        states[i+1] = ref_state
    
    return states
