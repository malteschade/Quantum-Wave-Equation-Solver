#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for plotting functions.}

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
# Other modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# -------- SETTINGS --------
plt.rcParams['font.family'] = 'Times New Roman'

# -------- CONSTANTS --------
ENUMS = ['a.)', 'b.)', 'c.)', 'd.)', 'e.)', 'f.)']
PATH_MULTIPLOT = './figures/forward_sim.png'

# -------- FUNCTIONS --------
def plot_multi(data, idx):
    """
    Plotting function for plotting multiple solvers at different time steps.
    
    Args:
        data (list): List of data dictionaries.
        idx (list): List of time indices to plot.
    
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    assert len(data) == 3, "Only three solvers supported"
    assert len(idx) == 5, "Please provide 5 time indices as idx for plotting"
    assert max(idx) < len(data[0]['times']), "idx out of range"

    # Read data (Assuming local = noise free, cloud = quantum computer !) TODO: Fix this
    settings = data[0]['settings']
    times, rho, mu = data[0]['times'], settings['rho'], settings['mu']
    bcs, nx = settings['bcs'], settings['nx']
    rho_lim = (1e3, 5e3)
    mu_lim = (0.5e10, 4.5e10)
    field_lim = (-1, 1)
    digits = 4

    solv_list = []
    for d in data:
        if d['settings']['solver'] == 'ode':
            solv_list.append('Classical ODE Solver')
        elif d['settings']['solver'] == 'exp':
            solv_list.append('Classical Exponential Solver')
        elif d['settings']['solver'] == 'local' and not d['settings']['backend']['fake']:
            solv_list.append('Noise Free Simulator')
        elif d['settings']['solver'] == 'local' and d['settings']['backend']['fake']:
            solv_list.append('Noise Model Simulator')
        elif (d['settings']['solver'] == 'cloud' and
              d['settings']['backend']['backend'] == 'ibmq_qasm_simulator'):
            solv_list.append('IBM QASM Simulator')
        elif (d['settings']['solver'] == 'cloud' and
              d['settings']['backend']['backend'] != 'ibmq_qasm_simulator'):
            solv_list.append('IBM Quantum Computer')

    # Prepare data with boundary conditions
    data_fields = []
    for d in data:
        field = np.zeros((len(times), nx + 2))
        field[:, 1:-1] = d['field']['u']
        if bcs['left'] == 'NBC':
            field[:, 0] = d['field']['u'][:, 0]
        if bcs['right'] == 'NBC':
            field[:, -1] = d['field']['u'][:, -1]
        data_fields.append(field)

    # Prepare medium
    medium_fields = []
    for d in [mu, rho]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        field[0] = d[0]
        field[-1] = d[-1]
        medium_fields.append(field)

    # Plot multiplot
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    ax_rho_mu = axes[0, 0]
    ax_mu = ax_rho_mu.twinx()

    ax_rho_mu.text(0.05, 0.9, ENUMS[0], transform=ax_rho_mu.transAxes,
                   fontsize=14)

    ax_rho_mu.plot(np.arange(nx+2), medium_fields[1], color='blue', label='$\\rho$')
    ax_rho_mu.set_ylabel('$\\rho$ [kg/m$^3$]', color='blue')
    ax_rho_mu.tick_params(axis='y', labelcolor='blue')
    ax_rho_mu.set_ylim(*rho_lim)

    ax_mu.plot(np.arange(nx+2), medium_fields[0], color='red', label='$\\mu$')
    ax_mu.set_ylabel('$\\mu$ [Pa]', color='red')
    ax_mu.tick_params(axis='y', labelcolor='red')
    ax_mu.set_ylim(*mu_lim)

    lines, labels = ax_rho_mu.get_legend_handles_labels()
    lines2, labels2 = ax_mu.get_legend_handles_labels()
    ax_mu.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_mu.set_xlabel('x [m]')
    ax_rho_mu.set_xlabel('x [m]')

    for i, t in enumerate(idx):
        ax = axes[(i+1) // 3, (i+1) % 3]
        ax.text(0.05, 0.9, ENUMS[i+1], transform=ax.transAxes,
                fontsize=14)
        for j, field in enumerate(data_fields):
            style = 'scatterplot' if j == 0 else 'lineplot'
            getattr(sns, style)(x=np.arange(nx+2), y=field[t], ax=ax,
                                label=solv_list[j],
                                color=['black', 'red', 'blue'][j])
        ax.set_title(f"t = {times[t]:.{digits}f} s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("u [$\\mu$ m]")
        ax.set_ylim(*field_lim)
        if i == 0:
            ax.legend(loc='lower right')
        else:
            ax.legend([],[], frameon=False)

    plt.tight_layout()
    plt.savefig(PATH_MULTIPLOT, dpi=300)
    plt.close(fig)
    return fig

def plot_medium(mu, rho, **kwargs):
    """
    Plotting function for plotting the medium.
    
    Args:
        mu (list): List of mu values.
        rho (list): List of rho values.
        
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    nx = len(rho)
    rho_lim = (1e3, 5e3)
    mu_lim = (0.5e10, 4.5e10)

    # Prepare medium
    medium_fields = []
    for d in [mu, rho]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        field[0] = d[0]
        field[-1] = d[-1]
        medium_fields.append(field)

    fig, axes = plt.subplots(1, figsize=(6, 3))
    ax_rho_mu = axes
    ax_mu = ax_rho_mu.twinx()

    ax_rho_mu.plot(np.arange(nx+2), medium_fields[1], color='blue', label='$\\rho$')
    ax_rho_mu.set_ylabel('$\\rho$ [kg/m$^3$]', color='blue')
    ax_rho_mu.tick_params(axis='y', labelcolor='blue')
    ax_rho_mu.set_ylim(*rho_lim)

    ax_mu.plot(np.arange(nx+2), medium_fields[0], color='red', label='$\\mu$')
    ax_mu.set_ylabel('$\\mu$ [Pa]', color='red')
    ax_mu.tick_params(axis='y', labelcolor='red')
    ax_mu.set_ylim(*mu_lim)

    lines, labels = ax_rho_mu.get_legend_handles_labels()
    lines2, labels2 = ax_mu.get_legend_handles_labels()
    ax_mu.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_mu.set_xlabel('x [m]')
    ax_rho_mu.set_xlabel('x [m]')

    plt.tight_layout()
    return fig

def plot_initial(u, v, bcs, **kwargs):
    """
    Plotting function for plotting the initial state.
    
    Args:
        u (list): List of u values.
        v (list): List of v values.
        bcs (dict): Boundary conditions.
    
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    nx = len(u)
    u_lim = (-1, 1)
    v_lim = (-1000, 1000)

    # Prepare initial state
    state_fields = []
    for d in [u, v]:
        field = np.zeros(nx+2)
        field[1:-1] = d if len(d) == nx else d[:-1]
        if bcs['left'] == 'NBC':
            field[0] = d[0]
        if bcs['right'] == 'NBC':
            field[-1] = d[-1]
        state_fields.append(field)

    fig, axes = plt.subplots(1, figsize=(6, 3))
    ax_u_v = axes
    ax_v = ax_u_v.twinx()

    ax_u_v.plot(np.arange(nx+2), state_fields[0], color='blue', label='u')
    ax_u_v.set_ylabel('u [$\\mu$m]', color='black')
    ax_u_v.tick_params(axis='y', labelcolor='black')
    ax_u_v.set_ylim(*u_lim)

    ax_v.plot(np.arange(nx+2), state_fields[1], color='red', label='v')
    ax_v.set_ylabel('v [$\\mu$m / s$^2$]', color='black')
    ax_v.tick_params(axis='y', labelcolor='black')
    ax_v.set_ylim(*v_lim)

    lines, labels = ax_u_v.get_legend_handles_labels()
    lines2, labels2 = ax_v.get_legend_handles_labels()
    ax_v.legend(lines + lines2, labels + labels2, loc='lower right')
    ax_v.set_xlabel('x [m]')
    ax_u_v.set_xlabel('x [m]')

    plt.tight_layout()
    return fig

def plot_error(data1, data2, **kwargs):
    """
    Plotting function for plotting the error between two data sets.
    
    Args:
        data1 (list): List of data values.
        data2 (list): List of data values.
        
    Returns:
        fig (matplotlib.pyplot.figure): Figure handle.
    """
    _ = kwargs
    e_lim = (1e-4, 1e+1)
    data1 = data1['field']['u']
    data2 = data2['field']['u']

    l2_errors = np.linalg.norm(data1 - data2, axis=1) / np.linalg.norm(data2, axis=1)
    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(l2_errors, color='blue')
    ax.axhline(np.mean(l2_errors), color='black', linestyle='--')
    ax.text(0.15, 0.9, f"Mean error: {np.mean(l2_errors):.2e}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Relative L2 error')
    ax.set_yscale('log')
    ax.set_ylim(*e_lim)
    ax.set_title('Relative L2 error over time')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.close(fig)
    return fig

# TODO: Fix plot returns/plottings
