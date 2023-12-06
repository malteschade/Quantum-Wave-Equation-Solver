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
# Own modules
from simulation.experiment import ForwardExperiment1D
from utility.distributions import spike, ricker, gaussian, raised_cosine, sinc, homogeneous, exponential, polynomial

# -------- FUNCTIONS --------
def main() -> None:
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

    # Define solvers
    experiment.add_solver('ode', **parameters)
    experiment.add_solver('local', **parameters)

    # Run experiment
    _ = experiment.run()

# -------- SCRIPT --------
if __name__ == '__main__':
    main()
