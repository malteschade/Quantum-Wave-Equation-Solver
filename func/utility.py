#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules

# other modules
import numpy as np
from scipy.linalg import sqrtm, inv
import plotly.express as px


__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'


def hamiltonian(nodes, m, k): # atm only for 2 nodes
    # Define utility matrices
    Z = np.zeros((nodes,nodes))
    I = np.identity(nodes)

    # Calculate operator D and square roots DS
    D = np.array(
        [[k[0]+k[1], -k[1]],
         [-k[1], k[0]+k[1]]])
    DS = sqrtm(D)
    INV_DS = sqrtm(inv(D))
    
    # Calculate inversion operator DV
    DV = np.block([[Z, I],[-D, Z]])
    
    # Calculate operator T and Hamiltonian H
    T = np.block([[DS, Z],[Z, 1j*I]])
    INV_T = np.block([[INV_DS, Z],[Z, -1j*I]])
    H = np.block([[Z, DS],[DS, Z]])
    
    return H, T, INV_T, DV


def prepare_state(state0, T):
    # Prepare initial state
    psi = T @ np.array(state0)
    norm = np.linalg.norm(psi)
    psi0 = psi / norm
    return psi0, norm


def plot_results(data, times, label=''):
    # Plot results
    fig = px.line(x=times, y=[col for col in data.T],
        title=f'Wave Field Simulation ({label})')
    return fig.show()
