#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from simulation.experiment import ForwardExperiment1D
from utility.distributions import spike, ricker, gaussian, raised_cosine, sinc, homogeneous, exponential, polynomial
from utility.plotting import plot_uv, plot_state, plot_medium, plot_heatmap

# -------- FUNCTIONS --------
def main():
    # Create experiment
    experiment = ForwardExperiment1D(verbose=2)

    # Define parameters
    nx = 63
    parameters = {
        'dx': 1,
        'nx': nx, 
        'dt': 0.001,
        'nt': 50,
        'order': 1,
        'bcs': {'left': 'DBC', 'right': 'DBC'},
        'mu': homogeneous(3e10, nx+1),
        'rho': homogeneous(3000, nx),
        'u': ricker(1, nx, nx//2, sigma=20),
        'v': homogeneous(0, nx),
        'backend': {}
        }

    #  Define solvers
    experiment.add_solver('ode', **parameters)
    experiment.add_solver('local', **parameters)

    # Run experiment
    _ = experiment.run()

# -------- SCRIPT --------
if __name__ == '__main__':
    main()
