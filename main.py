#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules
import warnings
import logging
import json
import pickle

# own modules
from func.utility import hamiltonian, prepare_state, process_results, plot_results
from func.simulation_classical import simulate_classical
from func.simulation_quantum import simulate_quantum


__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'


# -------- CONFIGURATION --------
path_settings = 'settings.json'


# -------- MAIN --------
def main():
    # Parse settings
    s = json.load(open(path_settings, 'r'))
    
    # Calculate Hamiltonian and dependent matrices
    H, T, INV_T, DV = hamiltonian(s['nodes'], s['m'], s['k'])

    
if __name__ == '__main__':
    main()
