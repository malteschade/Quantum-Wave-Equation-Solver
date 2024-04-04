#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Main module for running the quantum 1D elastic wave equation solver.}

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
# Own modules
from simulation.experiment import ForwardExperiment1D
from utility.distributions import (spike, ricker, gaussian, raised_cosine,
                                    sinc, homogeneous, exponential, polynomial)

# -------- FUNCTIONS --------
def main() -> None:
    """
    Runs the quantum 1D elastic wave equation solver.
    """

    # Create experiment
    experiment = ForwardExperiment1D(verbose=2)

    # Set Experiment Parameters
    nx = 7
    parameters = {
        'dx': 1,                                        # Grid spacing
        'nx': nx,                                       # Number of grid points
        'dt': 0.0001,                                   # Time stepping
        'nt': 19,                                       # Number of time steps
        'order': 1,                                     # Finite-difference order
        'bcs': {'left': 'DBC', 'right': 'DBC'},         # Boundary conditions
        'mu': raised_cosine(3e10, nx+1, nx, 6, 1e10),   # Elastic modulus distribution
        'rho': raised_cosine(2e3, nx, nx-1, 6, 2e3),    # Density distribution
        'u': spike(1, nx, nx//2+1),                     # Initial positions
        'v': homogeneous(0, nx),                        # Initial velocities
        'backend': {
            'synthesis': 'MatrixExponential',           # Time Evolution Synthesis Method
            'batch_size': 100,                          # Circuit Batch Size
            'fitter': 'cvxpy_gaussian',                 # State Tomography fitter
            'backend': 'ibmq_qasm_simulator',           # Cloud backend name
            'shots': 1000,                              # Number of circuit samples
            'optimization': 3,                          # Circuit optimization level
            'resilience': 1,                            # Circuit resilience level
            'seed': 0,                                  # Transpilation seed
            'local_transpilation': False,               # Local transpilation
            'method': 'statevector',                    # Classical simulation method
            'fake': None,                               # Fake backend model (Currently not supported)
            }
        }

    # Define solvers
    experiment.add_solver('ode', **parameters)
    experiment.add_solver('exp', **parameters)
    experiment.add_solver('local', **parameters)

    # Run experiment
    _ = experiment.run()

# -------- SCRIPT --------
if __name__ == '__main__':
    main()
