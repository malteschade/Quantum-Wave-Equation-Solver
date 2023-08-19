#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Forward Simulation of the 1D wave equation on quantum hardware.}
"""

# Other modules
from scipy.integrate import odeint

__author__ = '{Malte Leander Schade}'
__copyright__ = 'Copyright {2023}, {quantum_wave_simulation}'
__version__ = '{1}.{0}.{3}'
__maintainer__ = '{Malte Leander Schade}'
__email__ = '{mail@malteschade.com}'
__status__ = '{IN DEVELOPMENT}'


def simulate_classical(state0, times, DV):
    # Perform classical simulation
    states = odeint(lambda y, t: DV @ y, state0, times)
    return states
