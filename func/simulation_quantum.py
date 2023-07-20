#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# built-in modules
from itertools import product

# other modules
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate, SdgGate

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
             optimization=1, seed=0, max_time="5m"):
    
    # Determine number of necessary qubits
    num_qubits = int(np.log2(hamiltonian.shape[0]))
    
    # Determine state tomography bases
    bases = product("ZXY", repeat=num_qubits)


    # Prepare circuits
    circuit_groups = []
    for time in times:
        circuits = []
        for base in bases:
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
            for i, op in enumerate(base):
                qc.append(meas_circuits[op], [i])
            qc.barrier()
            
            # Measurement
            qc.measure(qr, cr)
        
            circuits.append(qc)
        circuit_groups.append(circuits)


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
            coupling_map = backend.configuration().coupling_map
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
            raise Exception("Invalid hardware model for local simulation.")
        
        # Run circuits in groups
        jobs = []
        for circuits in circuit_groups:
            jobs.append(sampler.run(circuits))

        # Get results
        result_groups = [job.result() for job in jobs]

    
    elif hardware == "ibmq":
        # Get Qiskit Runtime service and set options
        service = QiskitRuntimeService()
        options = Options(optimization_level=optimization, shots=shots)
        
        # Define cloud sampler primitive backend
        if model == "qasm_simulator":
            backend = service.backend("ibmq_qasm_simulator")
            
        elif model == "perth":
            backend = service.backend("ibm_perth")
            
        else:
            raise Exception("Invalid hardware model for cloud simulation.")
        
        # Run jobs in Session
        with Session(service=service, backend=backend, max_time=max_time) as session:
            # Initiate sampler
            sampler = Sampler(session=session, options=options)
            
            # Run circuits in groups
            jobs = []
            for circuits in circuit_groups:
                jobs.append(sampler.run(circuits))

            # Get results
            result_groups = [job.result() for job in jobs]
            
            # Close session
            session.close()
    
    return result_groups
