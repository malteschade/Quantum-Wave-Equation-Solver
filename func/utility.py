#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Forward Simulation of the 1D wave equation on quantum hardware.}
"""

# Other modules
import numpy as np
from scipy.linalg import cholesky, inv
import plotly.express as px
from qiskit.quantum_info import PauliList, SparsePauliOp


__author__ = '{Malte Leander Schade}'
__copyright__ = 'Copyright {2023}, {quantum_wave_simulation}'
__version__ = '{1}.{0}.{3}'
__maintainer__ = '{Malte Leander Schade}'
__email__ = '{mail@malteschade.com}'
__status__ = '{IN DEVELOPMENT}'


def _sparse_pauli(U):
    # Utility calculations
    nt = int(np.log2(U.shape[0]))
    n = 2**nt
    x = np.arange(n)
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(nt+1, dtype=x.dtype).reshape([1, nt+1])

    # Define Logical Pauli Operator Group 1
    B1_Z = (x & mask).astype(bool).reshape(xshape + [nt+1])
    B1_Z[:,-1] = ~B1_Z[:,-1]
    B1_X = np.zeros(B1_Z.shape, dtype=bool)
    B1_X[:,-1] = ~B1_X[:,-1]
    B1 = PauliList.from_symplectic(B1_Z, B1_X)

    # Define Logical Pauli Operator Group 2
    B2_Z = np.tile(B1_Z, (nt, 1))
    B2_X = np.concatenate([np.hstack([
        np.ones((n, nt-i), dtype=bool),
        np.zeros((n, i), dtype=bool),
        np.ones((n, 1), dtype=bool)
        ]) for i in range(nt)], axis=0)
    y_mask = (B2_Z & B2_X).sum(axis=1) % 2 == 0
    B2 = PauliList.from_symplectic(B2_Z[~y_mask], B2_X[~y_mask])

    # Define Logical Pauli Operator Group 3
    B2_Z[:,-1] = ~B2_Z[:,-1]
    B3 = PauliList.from_symplectic(B2_Z[y_mask], B2_X[y_mask])

    # Define Coefficient Sign Correction Masks        
    b2_mask = B2_X[~y_mask,0] & ((~B2_Z[~y_mask,0] & ~B2_X[~y_mask,1]) | (B2_Z[~y_mask,0] & B2_X[~y_mask,1]))
    b3_mask = B2_X[y_mask,0] & ((~B2_Z[y_mask,0] & ~B2_X[y_mask,1]) | (B2_Z[y_mask,0] & B2_X[y_mask,1]))

    # Coefficients of the Pauli decomposition
    B1_C = np.array([-np.sum(p.to_matrix(sparse=True).diagonal(k=0) * np.diagonal(U)) for p in B1.delete(-1, qubit=True)]) 
    B2_C = np.array([np.sum(p.to_matrix(sparse=True).diagonal(k=1) * np.diagonal(U, 1)) for p in B2.delete(-1, qubit=True)])
    B3_C = np.array([-1j*np.sum(p.to_matrix(sparse=True).diagonal(k=1) * np.diagonal(U, 1)) for p in B3.delete(-1, qubit=True)])

    # Sign correction (not correct for n=2 -> inverse masks there)
    if nt == 1:
        B2_C[~b2_mask] *= -1
        B3_C[~b3_mask] *= -1
    else:
        B2_C[b2_mask] *= -1
        B3_C[b3_mask] *= -1

    # Normalization
    B_C = np.concatenate([B1_C, B3_C, B2_C]) / 2**nt

    # Hamiltonian
    hamiltonian = SparsePauliOp(B1+B2+B3, B_C)
    return hamiltonian


def hamiltonian(m, k):
    # Calculate operator D
    D = np.diag(k[:-1]+k[1:]) - np.diag(k[1:-1], k=1) - np.diag(k[1:-1], k=-1) # last term not necessary for QC

    # Define utility matrices
    n = len(D)
    Z = np.zeros((n,n))
    I = np.identity(n)
    
    # Upper cholesky decomposition
    U = cholesky(D, lower=False)

    # Forward-transform T
    T = np.block([[U, Z],[Z, I]]) # large matrix formulation !

    # Back-transform INV_T
    INV_T = np.block([[inv(U),Z],[Z, I]]) # large matrix formulation !

    # Hamiltonian H
    H1 = _sparse_pauli(U) # Forward time
    #H1 = np.block([[Z, 1j*U],[-1j*U.T, Z]]) # Forward time
    #H2 = np.block([[Z, -1j*C],[1j*C.T, Z]]) # Backward time
    
    # Classical inversion operator DV
    DV = np.block([[Z, I],[-D, Z]])
    
    return H1, T, INV_T, DV


def prepare_state(state0, T):
    # Prepare initial state
    psi = T @ np.array(state0)
    norm = np.linalg.norm(psi)
    psi0 = psi / norm
    return psi0, norm


def plot_results(data, times, range_y, label=''):
    # Plot results
    fig = px.line(x=times, y=[col for col in data.T],
        title=f'Wave Field Simulation ({label})',
        range_y=range_y)
    return fig.show()
