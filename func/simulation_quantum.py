#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules
import warnings
import json
warnings.filterwarnings("ignore")
from time import sleep
from itertools import product

# other modules
import numpy as np
# import matplotlib as mpl
# mpl.use('WebAgg')

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, SdgGate

from qiskit_experiments.library.tomography.fitters import linear_inversion
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakePerth


__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'


# -------- MEASUREMENT CIRCUITS --------
# Z-meas rotation
meas_z = QuantumCircuit(1, name="PauliMeasZ")

# X-meas rotation
meas_x = QuantumCircuit(1, name="PauliMeasX")
meas_x.append(HGate(), [0])

# Y-meas rotation
meas_y = QuantumCircuit(1, name="PauliMeasY")
meas_y.append(SdgGate(), [0])
meas_y.append(HGate(), [0])

# Measurement circuits
meas_circuits = {"Z" : meas_z, "X" : meas_x, "Y" : meas_y}


# -------- QUANTUM SIMULATION --------
def simulate_quantum(psi0, hamiltonian, times, hardware, model="", shots=1000,
             optimization=1, resilience=1, seed=None, save_path=None):
    
    # Determine number of necessary qubits
    num_qubits = int(np.log2(hamiltonian.shape[0]))
    
    # Determine state tomography observables 
    observables = list(product("ZXY", repeat=num_qubits))


    # Prepare circuits
    print(f"Preparing circuits ...")
    n = 0
    circuit_groups = []
    for time in times:
        circuits = []
        for observable in observables:
            # Circuit definition
            qr = QuantumRegister(num_qubits)
            cr = ClassicalRegister(num_qubits)
            qc = QuantumCircuit(qr, cr)
            
            # Statevector preparation
            qc.prepare_state(psi0, qr)
            qc.barrier()
            
            # Time evolution
            qc.hamiltonian(operator=hamiltonian, time=time, qubits=list(qr))
            qc.barrier()
            
            # State Tomography Circuits
            for i, op in enumerate(observable):
                qc.append(meas_circuits[op], [i])
            qc.barrier()
            
            # Measurement
            qc.measure(qr, cr)

            n += 1
            circuits.append(qc)
        circuit_groups.append(circuits)
    print(f"{n} Circuits in {len(times)} jobs prepared ...")


    if hardware == "simulator":
        # Define local sampler primitive backend
        if model == "":
            backend = None
            sampler = AerSampler(
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed},
            )
        
        elif model == "perth":
            backend = FakePerth()
            coupling_map = backend.coupling_map
            noise_model = NoiseModel.from_backend(backend)
            
            sampler = AerSampler(
                backend_options={
                    "method": "density_matrix",
                    "coupling_map": coupling_map,
                    "noise_model": noise_model,
                },
                run_options={"seed": seed, "shots": shots},
                transpile_options={"seed_transpiler": seed},
            )

        else:
            raise Exception("Invalid model selection for local simulation.")
        
        print(f"Running circuits ...")
        # Run circuits in groups
        jobs = []
        for circuits in circuit_groups:
            jobs.append(sampler.run(circuits))

        # Get results
        result_groups = [job.result() for job in jobs]
        print(f"Circuits finished.")
    
    elif hardware == "ibmq":
        # Get Qiskit Runtime service and set options
        service = QiskitRuntimeService()
        options = Options(optimization_level=optimization,
                          resilience_level=resilience)
        
        # Define cloud sampler primitive backend
        if model == "qasm_simulator":
            backend = service.backend("ibmq_qasm_simulator")
            
        elif model == "perth":
            backend = service.backend("ibm_perth")
            
        else:
            raise Exception("Invalid model selection for cloud simulation.")
        
        # Run jobs in Session
        with Session(service=service, backend=backend) as session:
            # Initiate sampler
            sampler = Sampler(session=session, options=options)
            
            print(f"Running circuits ...")
            # Run circuits in groups
            jobs, job_ids = [], []
            for circuits in circuit_groups:
                job = sampler.run(circuits, shots=shots)
                job_ids.append(job.job_id())
                jobs.append(job)

            # Save job IDs
            json.dump(job_ids, open(save_path, "w"))
            
            # Wait for jobs to complete
            completed = False
            while completed == False:
                if all([job.status().name == "DONE" for job in jobs]):
                    completed = True
                sleep(30)
            
            # Get results
            result_groups = [job.result() for job in jobs]
            
            # Close session
            session.close()
        print(f"Circuits finished.")
    
    else:
        raise Exception("Invalid hardware selection.")
    
    return result_groups, observables


def load_job_ids(job_ids):
    # Get Qiskit Runtime service
    service = QiskitRuntimeService()
    
    # Load Jobs
    result_groups = []
    jobs = [service.job(job_id) for job_id in job_ids]
    
    # Check if all jobs are completed
    if not all([job.status().name == "DONE" for job in jobs]):
        raise Exception("Not all jobs are completed.")
    
    # Get results
    result_groups = [job.result() for job in jobs]
    
    return result_groups

# Run tomography
def run_tomography(measurements, observables, state0, psi0, norm0, INV_T):
    print("Running tomography...")
    # Define reference states and output array
    states = np.zeros((len(measurements)+1, len(state0)))
    states[0], ref_state = state0, state0
    
    for i, meas in enumerate(measurements):
        # Pad non-measured entries with zeros
        max_key = max(max(d.keys()) for d in meas.quasi_dists)
        [m.setdefault(i, 0) for m in meas.quasi_dists for i in range(max_key + 1)]
        
        # Read measurement frequencies
        freq = np.array([m[key] for m in meas.quasi_dists for key in range(max_key + 1)])
        freq = np.reshape(freq, (len(meas.quasi_dists), max_key+1))
        
        # Prepare data for tomography
        shot_data = np.array([m['shots'] for m in meas.metadata])
        outcome_data = np.expand_dims(freq * shot_data[:, np.newaxis], axis=0)
        replacement = {'Z': 0, 'X': 1, 'Y': 2}
        measurement_data = np.array([[replacement[char] for char in sublist] for sublist in observables])
        preparation_data = np.full((len(shot_data), 0), None)

        fitter_kwargs = {}
        fitter_kwargs["measurement_basis"] = PauliMeasurementBasis()
        fitter_kwargs["measurement_qubits"] = tuple(range(len(measurement_data[0])))

        # Run tomography
        rho, metadata = linear_inversion(outcome_data, shot_data, measurement_data, preparation_data, **fitter_kwargs)
        
        # Minimize phase difference with parallel transport (rho)
        values, vectors = np.linalg.eig(rho)
        psi_norm_2 = vectors[:, np.argmax(values)]
        
        # Minimize phase difference with parallel transport (psi)
        phase = np.imag(np.log(psi_norm_2.dot(np.conj(psi0))))
        psi_2 = np.exp(-1j * phase) * psi_norm_2 * norm0
        
        # Append state to states
        state = np.real(INV_T @ psi_2)

        # Minimize phase difference with parallel transport (state)
        phase = np.imag(np.log(state.dot(np.conj(ref_state))))
        ref_state = np.exp(-1j * phase) * state
        states[i+1] = ref_state
    
    print("Tomography finished.")
    
    return states[:-1]
