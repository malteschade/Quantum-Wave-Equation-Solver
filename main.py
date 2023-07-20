#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules
import os
import datetime
import warnings
import logging
import json
import pickle

# other modules
import numpy as np

# own modules
from func.utility import hamiltonian, prepare_state, plot_results
from func.simulation_classical import simulate_classical
from func.simulation_quantum import simulate_quantum, load_job_ids
from func.tomography import run_tomography


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
path_save = 'data'


# -------- MAIN --------
def main():
    # Parse settings
    s = json.load(open(path_settings, 'r'))
    
    # New simulation
    if s["simID"] == "":
        # Check settings
        if (len(s['m']) == len(s['node_pos']) == len(s['node_vel'])) > 0:
            pass
        else:
            raise ValueError('Parameters do not match.')
        
        if len(s['k']) == len(s['m'])+1:
            pass
        else:
            raise ValueError('Parameters do not match.')
    
        # Set up simulation ID location
        dt = datetime.datetime.now()
        simID = dt.strftime("%Y%m%dT%H%M%S")
        print(f'Starting simulation with ID: {simID}')
        os.makedirs(os.path.join(path_save, simID), exist_ok=True)
        
        # Save settings and log
        json.dump(s, open(os.path.join(path_save, simID, 'settings.json'), 'w'))
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(filename=os.path.join(path_save, simID, 'log.log'),
                            encoding='utf-8', level=logging.INFO,
                            format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger('qiskit_ibm_experiment').setLevel(logging.WARNING)
        
        # Calculate Hamiltonian and dependent matrices
        H, T, INV_T, DV = hamiltonian(s['nodes'], s['m'], s['k'])
        
        # Prepare initial state
        state0 = s['node_pos'] + s['node_vel']
        psi0, norm0 = prepare_state(state0, T)
        
        # Define time steps
        dt, steps = s['dt'], s['steps']
        times = np.arange(dt, dt*(steps+1), dt)

        # Simulate classical
        states_cl = simulate_classical(state0, times, DV)
        np.savetxt(os.path.join(path_save, simID, 'sim_classical.csv'), states_cl, delimiter=",")
        
        # Save metadata
        metadata = {"times" : times.tolist(), "state0" : state0, "psi0" : psi0.tolist(), "norm0" : norm0,
                    "H" : H.tolist(), "T" : T.tolist(), "INV_T" : INV_T.tolist(), "DV" : DV.tolist()}
        pickle.dump(metadata, open(os.path.join(path_save, simID, 'metadata.pkl'), 'wb'))
        
        # Simulate quantum
        job_id_path = os.path.join(path_save, simID, 'job_ids.json')
        measurements, observables = simulate_quantum(psi0, H, times, s['hardware'], s['model'], s['shots'],
                                        s['optimization'], s['resilience'], s['seed'], save_path=job_id_path)
        pickle.dump(measurements, open(os.path.join(path_save, simID, 'measurements.pkl'), 'wb'))
        json.dump(observables, open(os.path.join(path_save, simID, 'observables.json'), 'w'))
        print('New simulation finished.')
    
    
    # Load existing simulation
    else:
        print(f'Loading simulation with ID: {s["simID"]}')
        if not os.path.exists(os.path.join(path_save, s["simID"])):
            raise ValueError(f'Simulation ID does not exist.')
        
        if os.path.exists(os.path.join(path_save, s["simID"], 'measurements.pkl')):
            print("Reading measurements from file.")
            measurements = pickle.load(open(os.path.join(path_save, s["simID"], 'measurements.pkl'), 'rb'))
        
        elif os.path.exists(os.path.join(path_save, s["simID"], 'job_ids.json')):
            print("Loading measurements from IBMQ.")
            job_id_path = os.path.join(path_save, s["simID"], 'job_ids.json')
            job_ids = json.load(open(job_id_path, 'r'))
            measurements = load_job_ids(job_ids)
            pickle.dump(measurements, open(os.path.join(path_save, s["simID"], 'measurements.pkl'), 'wb'))
        print('Existing simulation loaded.')
        
        # Load variables
        metadata = pickle.load(open(os.path.join(path_save, s["simID"], 'metadata.pkl'), 'rb'))
        times = metadata["times"]
        states_cl = np.loadtxt(os.path.join(path_save, s["simID"], 'sim_classical.csv'), delimiter=",")
        observables = json.load(open(os.path.join(path_save, s["simID"], 'bases.json'), 'r'))
        
    
    # Process quantum results
    states_qc = run_tomography(measurements, observables, state0, psi0, norm0, INV_T)
    
    
    # Plot results
    plot_results(states_cl, times, 'classical')
    
    
    
if __name__ == '__main__':
    main()
