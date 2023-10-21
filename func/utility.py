#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Forward Simulation of the 1D wave equation on quantum hardware.}
"""

# Other modules
import numpy as np
from scipy.linalg import cholesky, inv
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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

def gaussian_blob(x, mu=0, sigma=1, c=0, s=1):
    y = np.exp(-((x - mu)**2) / (2*sigma**2)) * s + c
    return y

def raised_cosine(x, c=0, s=1):
    return 0.5 * (1 + np.cos(np.pi * x)) * np.where(np.abs(x) < 1, 1, 0) * s + c

def hamiltonian(m, k):
    #k = gaussian_blob(k, mu=5, sigma=1.1, c=1, s=10)
    
    # Raised Cosine (Partial domain)
    k1 = np.linspace(-0.9, 0, int((len(k)-1)/2)+2)
    k1 = raised_cosine(k1, 2e10, 1e10) 
    k2 = [2e10]*(int((len(k)-1)/2)-2)
    k = np.concatenate([k2, k1, [0]])
    print(k)
    
    m1 = np.linspace(-0.9, 0, int(len(m)/2)+2)
    m1 = raised_cosine(m1, 2000, 1000) 
    m2 = [2000]*(int(len(m)/2)-2) 
    m = np.concatenate([m2, m1])
    print(m)
    
    #k = k[::-1]
    
    #k = [1000] * int((len(k)-1)/2) + [500] * int((len(k)-1)/2) + [0] # Step Function
    #k = (np.linspace(1, 2, len(k)) * 500).tolist() # Linear Increase
    
    # Calculate operator D
    dx = 1 # space sampling
    SQRT_M = np.diag(1/np.sqrt(m))
    MU = (1/dx**2)*SQRT_M@np.diag(k[:-1])@SQRT_M
    mu = MU.diagonal()

    # Finite difference
    FD1 = (
        np.diag([-1/2]*(len(k)-2), k=-1) + 
        np.diag([1/2]*(len(k)-2), k=1)
    )
    FD1[0,0] = -0.5 # Forward difference ? (Boundary)
    FD1[-1,-1] = 0.5 # Backward difference ? (Boundary)
    
    FD2 = (
        np.diag([1]*(len(k)-2), k=-1) + 
        np.diag([-2]*(len(k)-1), k=0) +
        np.diag([1]*(len(k)-2), k=1)
    )
    print(FD1)
    
    # Calculate operator D (Finite Difference)
    D1 = np.diag(np.dot(FD1, mu)) @ FD1
    D2 = MU @ FD2
    D = -(D1 + D2)
    
    print(D1)
    print(D2)
    print(D)
    #D = -D2
    
    #D = np.diag(mu[:-1], k=-1) - np.diag(mu[1:], k=1)
    #D = np.diag(-mu[1:], k=-1) + np.diag(2*mu, k=0) + np.diag(-mu[:-1], k=1) # equivalent to above solution
    
    # Calculate operator D (Mass-Spring System)
    #D = np.diag(k[:-1]+k[1:]) - np.diag(k[1:-1], k=1) - np.diag(k[1:-1], k=-1)

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
    #H1 = _sparse_pauli(U) # Forward time
    H1 = np.block([[Z, 1j*U],[-1j*U.T, Z]]) # Forward time
    
    #H2 = np.block([[Z, -1j*C],[1j*C.T, Z]]) # Backward time
    
    # print(H1)
    # H1[0,0] = 1
    # H1[0,1] = 0
    # H1[1,0] = 0
    
    # H1[-1,-1] = 1
    # H1[-1,-2] = 0
    # H1[-2,-1] = 0
    # print(H1)
    
    # Classical inversion operator DV
    DV = np.block([[Z, I],[-D, Z]])
    
    return H1, T, INV_T, DV

def hamiltonian_large(m_l, m_v, k_l, k_v):
    # Get m and k
    m = np.full(m_l, m_v)
    k = np.full(k_l, k_v)
    
    # Calculate operator D
    M = np.diag(m)
    SQRT_M = np.diag(np.sqrt(m))
    MU = SQRT_M@np.diag(k[:-1])@SQRT_M
    mu = SQRT_M*k[:-1]*SQRT_M
    print(mu)
    print(MU)
    raise Exception
    
    # Finite difference
    FD1 = (
        np.diag([-1/2]*(len(k)-2), k=-1) + 
        np.diag([1/2]*(len(k)-2), k=1)
    )
    
    FD2 = (
        np.diag([1]*(len(k)-2), k=-1) + 
        np.diag([-2]*(len(k)-1), k=0) +
        np.diag([1]*(len(k)-2), k=1)
    )
    
    # Calculate operator D (Finite Difference)
    D = -(np.diag(np.dot(FD1, mu))@FD1 + MU @ FD2)

    # Calculate operator D (Mass-Spring System)
    #D = np.diag(k[:-1]+k[1:]) - np.diag(k[1:-1], k=1) - np.diag(k[1:-1], k=-1)
    
    #print(D)
    #print(np.linalg.eigvals(D))
    
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
    #H1 = _sparse_pauli(U) # Forward time (Mass-Spring System)
    H1 = np.block([[Z, 1j*U],[-1j*U.T, Z]]) # Forward time
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


def plot_multisub(data, data2, specific_time_indices, time):
    dt = 0.0002
    
    midpoint = data.shape[1] // 2
    data = data[:, :midpoint]
    data2 = data2[:, :midpoint]
    
    zeros = np.zeros((1, data.shape[0]))
    data = np.insert(data, 0, zeros, axis=1)
    data = np.insert(data, data.shape[1], zeros, axis=1)

    data2 = np.insert(data2, 0, zeros, axis=1)
    data2 = np.insert(data2, data2.shape[1], zeros, axis=1)

    l2_errors = np.linalg.norm(data - data2, axis=1) / np.linalg.norm(data2, axis=1)
    
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[f"$$t={idx*dt:.4f}s$$" for idx in specific_time_indices] + [f"$$Error:\;\overline E  ={l2_errors.mean():.4f} $$"])
    
    for plot_idx, time_idx in enumerate(specific_time_indices):
        show_legend = plot_idx == 0
        fig.add_trace(go.Scatter(x=list(range(midpoint+2)),
                                 y=data[time_idx],
                                 mode='lines',
                                 name='Quantum',
                                 line=dict(color='blue'),
                                 showlegend=show_legend),
                      row=plot_idx//3 + 1, col=plot_idx%3 + 1)

        fig.add_trace(go.Scatter(x=list(range(midpoint+2)),
                                 y=data2[time_idx],
                                 mode='markers',
                                 name='Classical',
                                 marker=dict(color='red'),
                                 showlegend=show_legend),
                      row=plot_idx//3 + 1, col=plot_idx%3 + 1)
        #tickvals = np.array([0, 0.25, 0.5, 0.75, 1]) * (len(data[time_idx])+1)
        #ticklabels = ['-0.5', '-0.25', '0', '0.25', '0.5']
        #fig.update_xaxes(title_text="X", row=plot_idx//3 + 1, col=plot_idx%3 + 1, tickvals=tickvals, ticktext=ticklabels)
        fig.update_xaxes(title_text="$Distance [m]$", row=plot_idx//3 + 1, col=plot_idx%3 + 1)
        fig.update_yaxes(title_text="$Amplitude [\mu m]$", row=plot_idx//3 + 1, col=plot_idx%3 + 1)
        
    fig.add_trace(go.Scatter(x=list(range(len(l2_errors))), y=l2_errors, mode='lines', name='L2 Norm', line=dict(color='black'), showlegend=False), row=2, col=3)
    tickvals = np.array([0, 5, 10, 15])
    ticklabels = [f'{idx*dt:.3f}' for idx in tickvals]
    fig.update_xaxes(title_text="$Time [s]$", tickvals=tickvals, ticktext=ticklabels, row=2, col=3)
    #fig.update_yaxes(title_text="$log_{10}(||u^q - u^c|| / ||u^c||)$", type='log', row=2, col=3)
    fig.update_yaxes(title_text="$||u^q - u^c|| / ||u^c||$", row=2, col=3)
    
    fig.update_yaxes(range=[-1, 1])
    #fig.update_yaxes(row=2, col=3, range=[-3, 1])
    fig.update_yaxes(row=2, col=3, range=[0, 3])
    
    fig.update_layout(font=dict(size=15))
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=20)
        annotation['yshift'] = 10
    
    fig.write_image("forward_sim.png", scale=4, width=1400, height=800)
    return fig.show()

