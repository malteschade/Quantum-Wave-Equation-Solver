#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Forward Simulation of the 1D wave equation on quantum hardware.}
"""

# Built-in modules
import os
import datetime
import logging
import json
import pickle

# Other modules
import numpy as np

# Own modules
from func.utility import hamiltonian, prepare_state, plot_results
from func.simulation_classical import simulate_classical
from func.simulation_quantum import simulate_quantum, load_job_ids, run_tomography


__author__ = '{Malte Leander Schade}'
__copyright__ = 'Copyright {2023}, {quantum_wave_simulation}'
__version__ = '{1}.{0}.{3}'
__maintainer__ = '{Malte Leander Schade}'
__email__ = '{mail@malteschade.com}'
__status__ = '{IN DEVELOPMENT}'


# -------- CONFIGURATION --------
path_settings = 'settings.json'
path_save = 'data'


# -------- MAIN --------
def main():
    # Parse settings
    s = json.load(open(path_settings, 'r'))
    
    # Start new simulation
    if s["simID"] == "":
        # Check settings
        if not (len(s['m']) == len(s['node_pos']) == len(s['node_vel'])) > 0:
            raise ValueError('Initial parameters not valid.')
        if len(s['k']) != len(s['m'])+1:
            raise ValueError('Initial parameters not valid.')
    
        # Set up simulation ID location
        dt = datetime.datetime.now()
        simID = dt.strftime("%Y%m%dT%H%M%S")

        print(f'Starting simulation with ID: {simID}')
        os.makedirs(os.path.join(path_save, simID), exist_ok=True)

        # Save settings and log
        json.dump(s, open(os.path.join(path_save, simID, 'settings.json'), 'w'))

        # Setup logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            filename=os.path.join(path_save, simID, 'log.log'),
            encoding='utf-8',
            level=logging.INFO,
            format=log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger('qiskit_ibm_experiment').setLevel(logging.WARNING)

        # Calculate Hamiltonian and dependent matrices
        H, T, INV_T, DV = hamiltonian(np.array(s['m']), np.array(s['k']))

        # Prepare initial state
        state0 = s['node_pos'] + s['node_vel']
        psi0, norm0 = prepare_state(state0, T)

        # Define time steps
        dt, steps = s['dt'], s['steps']
        times = np.arange(dt, dt*(steps+1), dt)

        # Simulate classical ODE
        print('Simulating classical...')
        states_cl = simulate_classical(state0, times, DV)
        np.savetxt(os.path.join(path_save, simID, 'sim_classical.csv'), states_cl, delimiter=",")

        print('Classical simulation finished.')

        # Save simulation metadata
        metadata = {
            "times" : times.tolist(),
            "state0" : state0,
            "psi0" : psi0.tolist(),
            "norm0" : norm0,
            #"H" : H.tolist(),
            "H": H,
            "T" : T.tolist(),
            "INV_T" : INV_T.tolist(),
            "DV" : DV.tolist()
        }
        pickle.dump(metadata, open(os.path.join(path_save, simID, 'metadata.pkl'), 'wb'))

        # Simulate quantum
        print('Simulating quantum...')

        # Define job ID path and observables path
        job_id_path = os.path.join(path_save, simID, 'job_ids.json')
        obs_path = os.path.join(path_save, simID, 'observables.json')
        
        # Simulate quantum Hamiltonian time evolution
        measurements, observables = simulate_quantum(
            psi0, 
            H, 
            times, 
            s['hardware'], 
            s['model'], 
            s['shots'],
            s['optimization'], 
            s['resilience'], 
            s['seed'], 
            job_id_path=job_id_path,
            obs_path = obs_path
        )

        # Save measurements
        pickle.dump(measurements, open(os.path.join(path_save, simID, 'measurements.pkl'), 'wb')) 

        print('Quantum simulation finished.')

    # Load existing simulation
    else:
        print(f'Loading simulation with ID: {s["simID"]}')

        # Check if simulation ID exists
        if not os.path.exists(os.path.join(path_save, s["simID"])):
            raise ValueError(f'Simulation ID does not exist.')
        
        # Load measurements
        if os.path.exists(os.path.join(path_save, s["simID"], 'measurements.pkl')):
            print("Reading measurements from file.")
            measurements = pickle.load(open(os.path.join(path_save, s["simID"], 'measurements.pkl'), 'rb'))
        
        # Load measurements from IBMQ
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
        state0 = metadata["state0"]
        psi0 = np.array(metadata["psi0"])
        norm0 = metadata["norm0"]
        INV_T = np.array(metadata["INV_T"])
        states_cl = np.loadtxt(os.path.join(path_save, s["simID"], 'sim_classical.csv'), delimiter=",")
        observables = json.load(open(os.path.join(path_save, s["simID"], 'observables.json'), 'r'))
        
    # Process quantum results
    states_qc = run_tomography(measurements, observables, state0, psi0, norm0, INV_T)

    # Plot results
    r_y = np.max(np.abs(states_cl))*1.05
    plot_results(states_cl, times, [-r_y, r_y], 'classical')
    plot_results(states_qc, times, [-r_y, r_y], 'quantum')
    plot_results((states_cl-states_qc)/np.max(states_cl), times, [-1, 1], 'relative difference')
    print(f'Mean Absolute Error: {np.mean(np.abs(states_cl-states_qc))}')
    print(f'Mean Relative error: {np.mean(np.abs((states_cl-states_qc)/np.max(states_cl)))}')


if __name__ == '__main__':
    main()
