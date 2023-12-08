#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Main module for running the quantum 1D elastic wave equation solver.}

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
            'fake': None,                               # Fake backend model
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
