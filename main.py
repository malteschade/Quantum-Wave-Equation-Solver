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
from func.utility import hamiltonian, prepare_state, hamiltonian_large, plot_multisub
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
log_name = 'log.log'
meta_name = 'metadata.pkl'
simcl_name = 'sim_classical.csv'
jobid_name = 'job_ids.json'
obs_name = 'observables.json'
meas_name = 'measurements.pkl'


# -------- MAIN --------
def main():
    # Parse settings
    s = json.load(open(path_settings, 'r'))
    
    # Start new simulation
    if s["simID"] == "":
        # Check settings
        # if not (len(s['m']) == len(s['node_pos']) == len(s['node_vel'])) > 0:
        #     raise ValueError('Initial parameters not valid.')
        # if len(s['k']) != len(s['m'])+1:
        #     raise ValueError('Initial parameters not valid.')
    
        # Set up simulation ID location
        dt = datetime.datetime.now()
        simID = dt.strftime("%Y%m%dT%H%M%S")

        print(f'Starting simulation with ID: {simID}')
        os.makedirs(os.path.join(path_save, simID), exist_ok=True)

        # Save settings and log
        json.dump(s, open(os.path.join(path_save, simID, path_settings), 'w'))

        # Setup logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            filename=os.path.join(path_save, simID, log_name),
            encoding='utf-8',
            level=logging.INFO,
            format=log_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger('qiskit_ibm_experiment').setLevel(logging.WARNING)

        if s['large_sim'] == True:
            # Calculate Hamiltonian and dependent matrices
            H, T, INV_T, DV = hamiltonian_large(s['m_l'], s['m_v'], s['k_l'], s['k_v'])
            
            # Define initial conditions
            if s['initial'] == "start_pos_spike":
                state0 = np.concatenate([np.array([1]), np.full(2*s['m_l']-1, 0)])
                psi0, norm0 = prepare_state(state0, T)
            
            
        elif s['large_sim'] == False:
            # Calculate Hamiltonian and dependent matrices
            H, T, INV_T, DV = hamiltonian(np.array(s['m']), np.array(s['k']))
            
            # Prepare initial state
            #state0 = s['node_pos'] + s['node_vel']
            
            # Ricker wavelet
            # n = len(s['node_pos'])
            # f = 15
            # rdt = 0.02
            # t = np.arange(0, n) * rdt - ((n * rdt) / 2.0)  # Centering the wavelet
            # wavelet = (1 - 2 * np.pi**2 * f**2 * t**2) * np.exp(-np.pi**2 * f**2 * t**2)
            # state0 = wavelet.tolist() + s['node_vel']
            
            # Spike wavelet
            state0 = np.concatenate([np.array([1]), np.full(len(s['node_pos'])*2-1, 0)])
            
            psi0, norm0 = prepare_state(state0, T)

        # Define time steps
        dt, steps = s['dt'], s['steps']
        times = np.arange(dt, dt*(steps+1), dt)

        # Simulate classical ODE
        print('Simulating classical...')
        states_cl = simulate_classical(state0, times, DV)
        np.savetxt(os.path.join(path_save, simID, simcl_name), states_cl, delimiter=",")

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
        pickle.dump(metadata, open(os.path.join(path_save, simID, meta_name), 'wb'))

        # Simulate quantum
        print('Simulating quantum...')

        # Define job ID path and observables path
        job_id_path = os.path.join(path_save, simID, jobid_name)
        obs_path = os.path.join(path_save, simID, obs_name)
        
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
        pickle.dump(measurements, open(os.path.join(path_save, simID, meas_name), 'wb')) 

        print('Quantum simulation finished.')

    # Load existing simulation
    else:
        print(f'Loading simulation with ID: {s["simID"]}')

        # Check if simulation ID exists
        if not os.path.exists(os.path.join(path_save, s["simID"])):
            raise ValueError(f'Simulation ID does not exist.')
        
        # Load measurements
        if os.path.exists(os.path.join(path_save, s["simID"],meas_name)):
            print("Reading measurements from file.")
            measurements = pickle.load(open(os.path.join(path_save, s["simID"], meas_name), 'rb'))
        
        # Load measurements from IBMQ
        elif os.path.exists(os.path.join(path_save, s["simID"], jobid_name)):
            print("Loading measurements from IBMQ.")
            job_id_path = os.path.join(path_save, s["simID"], jobid_name)
            job_ids = json.load(open(job_id_path, 'r'))
            measurements = load_job_ids(job_ids)
            pickle.dump(measurements, open(os.path.join(path_save, s["simID"], meas_name), 'wb'))

        print('Existing simulation loaded.')
        
        # Load variables
        metadata = pickle.load(open(os.path.join(path_save, s["simID"], meta_name), 'rb'))
        times = metadata["times"]
        state0 = metadata["state0"]
        psi0 = np.array(metadata["psi0"])
        norm0 = metadata["norm0"]
        INV_T = np.array(metadata["INV_T"])
        states_cl = np.loadtxt(os.path.join(path_save, s["simID"], simcl_name), delimiter=",")
        observables = json.load(open(os.path.join(path_save, s["simID"], obs_name), 'r'))
        
    # Process quantum results
    states_qc = run_tomography(measurements, observables, state0, psi0, norm0, INV_T)

    # Plot results
    time_idx = [0, 4, 8, 12, 16]
    plot_multisub(states_qc, states_cl, time_idx, len(times))
    print(f'Mean Absolute Error: {np.mean(np.abs(states_cl-states_qc))}')
    print(f'Mean Relative error: {np.mean(np.abs((states_cl-states_qc)/np.max(states_cl)))}')


if __name__ == '__main__':
    main()
