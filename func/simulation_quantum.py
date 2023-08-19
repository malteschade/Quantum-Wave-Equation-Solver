#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Forward Simulation of the 1D wave equation on quantum hardware.}
"""

# Built-in modules
import json
from time import sleep
from itertools import product

# Other modules
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, SdgGate
from qiskit_experiments.library.tomography.fitters import linear_inversion, cvxpy_linear_lstsq, cvxpy_gaussian_lstsq
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakePerth

__author__ = '{Malte Leander Schade}'
__copyright__ = 'Copyright {2023}, {quantum_wave_simulation}'
__version__ = '{1}.{0}.{3}'
__maintainer__ = '{Malte Leander Schade}'
__email__ = '{mail@malteschade.com}'
__status__ = '{IN DEVELOPMENT}'


# -------- CONFIGURATION --------
# Define measurement circuits
def define_measurement_circuits():
    meas_z = QuantumCircuit(1, name="PauliMeasZ")  # Z-meas rotation
    meas_x = QuantumCircuit(1, name="PauliMeasX")  # X-meas rotation
    meas_x.append(HGate(), [0])
    meas_y = QuantumCircuit(1, name="PauliMeasY")  # Y-meas rotation
    meas_y.append(SdgGate(), [0])
    meas_y.append(HGate(), [0])

    return {"Z" : meas_z, "X" : meas_x, "Y" : meas_y}

meas_circuits = define_measurement_circuits()


# -------- MAIN --------
def simulate_quantum(psi0, hamiltonian, times, hardware, model="", shots=1000,
             optimization=1, resilience=1, seed=None, job_id_path=None, obs_path=None):
    
    num_qubits = int(np.log2(len(psi0)))
    observables = list(product("ZXY", repeat=num_qubits)) # full observable basis
    
    # Save observables
    json.dump(observables, open(obs_path, 'w'))

    circuit_groups = _prepare_circuits(times, observables, psi0, hamiltonian, num_qubits)

    if hardware == "simulator":
        result_groups = _run_simulator(circuit_groups, model, seed, shots)
        
    elif hardware == "ibmq":
        result_groups = _run_ibmq(circuit_groups, model, optimization, resilience, shots, job_id_path)
    
    else:
        raise Exception("Invalid hardware selection.")
    
    return result_groups, observables


def load_job_ids(job_ids):
    runtime_service = QiskitRuntimeService()
    
    jobs = [runtime_service.job(job_id) for job_id in job_ids]
    
    _ensure_jobs_completed(jobs)
    
    results = [job.result() for job in jobs]
    
    return results


def run_tomography(measurements, observables, initial_state, initial_psi, norm0, INV_T):
    print("Running tomography...")
    
    states = np.zeros((len(measurements)+1, len(initial_state)))
    states[0], ref_state = initial_state, initial_state
    
    for i, measurement in enumerate(measurements):
        max_key = _get_max_key(measurement)
        _pad_with_zeros(measurement, max_key)

        freq = _get_frequencies(measurement, max_key)
        outcome_data, shot_data = _prepare_tomography_data(measurement, freq)

        measurement_data = _get_measurement_data(observables)
        preparation_data = np.full((len(shot_data), 0), None)

        rho = _run_inversion(measurement_data, outcome_data, shot_data, preparation_data)
        
        psi_2 = _minimize_phase_difference(rho, initial_psi, norm0)
        
        state = np.real(INV_T @ psi_2)
        ref_state = _minimize_state_phase_difference(state, ref_state)
        
        states[i+1] = np.real(ref_state)
    
    print("Tomography finished.")
    
    return states[:-1]


# -------- FUNCTIONS --------
def _prepare_circuits(times, observables, psi0, hamiltonian, num_qubits):
    print(f"Preparing circuits ...")
    n = 0
    circuit_groups = []
    for time in times:
        circuits = []
        for observable in observables:
            circuit = _prepare_single_circuit(num_qubits, psi0, hamiltonian, time, observable)
            circuits.append(circuit)
            n += 1
        circuit_groups.append(circuits)
    print(f"{n} Circuits in {len(times)} jobs prepared ...")
    return circuit_groups

def _prepare_single_circuit(num_qubits, psi0, hamiltonian, time, observable):
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
    return qc

def _run_simulator(circuit_groups, model, seed, shots):
    sampler = _get_simulator_sampler(model, seed, shots)
    print(f"Running circuits ...")
    jobs = [sampler.run(circuits) for circuits in circuit_groups]
    _wait_for_jobs_to_complete(jobs)
    result_groups = [job.result() for job in jobs]
    print(f"Circuits finished.")
    return result_groups

def _get_simulator_sampler(model, seed, shots):
    if model == "":
        backend = None
        sampler = AerSampler(
            backend_options={
                "method": "density_matrix",
                "max_parallel_experiments": 0
            },
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed})
    elif model == "perth":
        backend = FakePerth()
        coupling_map = backend.coupling_map
        noise_model = NoiseModel.from_backend(backend)
        sampler = AerSampler(
            backend_options={
                "method": "density_matrix",
                "coupling_map": coupling_map,
                "noise_model": noise_model,
                "max_parallel_experiments": 0
            },
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed},
        )
    else:
        raise Exception("Invalid model selection for local simulation.")
    return sampler

def _run_ibmq(circuit_groups, model, optimization, resilience, shots, job_id_path):
    service = QiskitRuntimeService()
    options = Options(optimization_level=optimization,
                      resilience_level=resilience)
    backend = _get_ibmq_backend(service, model)
    with Session(service=service, backend=backend) as session:
        sampler = Sampler(session=session, options=options)
        print(f"Running circuits ...")
        jobs, job_ids = [], []
        for i, circuits in enumerate(circuit_groups):
            print(f"Transmitting job {i+1}/{len(circuit_groups)}")
            job_transmitted = False
            while job_transmitted == False:
                try:
                    job = sampler.run(circuits, shots=shots)
                    job_transmitted = True
                except Exception as e:
                    print(e)
                    print('Retrying in 10 seconds ...')
                    sleep(10)
            job_ids.append(job.job_id())
            jobs.append(job)
        # Save job IDs
        json.dump(job_ids, open(job_id_path, "w"))
        _wait_for_jobs_to_complete(jobs)
        result_groups = [job.result() for job in jobs]
        session.close()
    print(f"Circuits finished.")
    return result_groups

def _get_ibmq_backend(service, model):
    if model == "qasm_simulator":
        backend = service.backend("ibmq_qasm_simulator")
    elif model == "perth":
        backend = service.backend("ibm_perth")
    elif model == "belem":
        backend = service.backend("ibmq_belem")
    else:
        raise Exception("Invalid model selection for cloud simulation.")
    return backend

def _wait_for_jobs_to_complete(jobs):
    completed = False
    while not completed:
        done_list = [job.status().name == "DONE" for job in jobs]
        print(f"Jobs completed: {sum(done_list)}/{len(done_list)}")
        if all(done_list):
            completed = True
        sleep(10)

def _ensure_jobs_completed(jobs):
    if not all([job.status().name == "DONE" for job in jobs]):
        raise Exception("Not all jobs are completed.")

def _get_max_key(measurement):
    return max(max(dist.keys()) for dist in measurement.quasi_dists)

def _pad_with_zeros(measurement, max_key):
    [dist.setdefault(i, 0) for dist in measurement.quasi_dists for i in range(max_key + 1)]

def _get_frequencies(measurement, max_key):
    freq = np.array([dist[key] for dist in measurement.quasi_dists for key in range(max_key + 1)])
    return np.reshape(freq, (len(measurement.quasi_dists), max_key+1))

def _prepare_tomography_data(measurement, freq):
    shot_data = np.array([meta['shots'] for meta in measurement.metadata])
    outcome_data = np.expand_dims(freq * shot_data[:, np.newaxis], axis=0)
    return outcome_data, shot_data

def _get_measurement_data(observables):
    replacement = {'Z': 0, 'X': 1, 'Y': 2}
    return np.array([[replacement[char] for char in sublist] for sublist in observables])

def _run_inversion(measurement_data, outcome_data, shot_data, preparation_data):
    fitter_kwargs = {
        "measurement_basis": PauliMeasurementBasis(),
        "measurement_qubits": tuple(range(len(measurement_data[0])))
    }
    #rho, metadata = linear_inversion(outcome_data, shot_data, measurement_data, preparation_data, **fitter_kwargs)
    rho, metadata = cvxpy_gaussian_lstsq(outcome_data, shot_data, measurement_data, preparation_data, **fitter_kwargs)
    #rho, metadata = cvxpy_linear_lstsq(outcome_data, shot_data, measurement_data, preparation_data, **fitter_kwargs)
    
    return rho

def _minimize_phase_difference(rho, initial_psi, norm0):
    values, vectors = np.linalg.eig(rho)
    psi_norm_2 = vectors[:, np.argmax(values)]
    phase = np.imag(np.log(psi_norm_2.dot(np.conj(initial_psi))))
    return np.exp(-1j * phase) * psi_norm_2 * norm0

def _minimize_state_phase_difference(state, ref_state):
    phase = np.imag(np.log(state.dot(np.conj(ref_state))))
    return np.exp(-1j * phase) * state
