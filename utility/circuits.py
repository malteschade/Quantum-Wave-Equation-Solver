from itertools import product

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.synthesis import SuzukiTrotter, LieTrotter, QDrift, MatrixExponential

SYNTHESIS = {
    'MatrixExponential': MatrixExponential(),
    'LieTrotter': LieTrotter(reps=3),
    'SuzukiTrotter': SuzukiTrotter(order=2, reps=2),
    'QDrift': QDrift(reps=4)
}

class CircuitGen1DA:
    def __init__(self):
        self.meas_circuits = self.measurement_circuits()
        
    def tomography_circuits(self, initial_state, hamiltonian, times,
                         synthesis='MatrixExponential', batch_size=100):
        self.num_qubits = int(np.log2(hamiltonian.shape[0]))
        self.observables = list(product("ZX", repeat=self.num_qubits))
        synthesis = SYNTHESIS[synthesis]
        op = SparsePauliOp.from_operator(Operator(hamiltonian))
        
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        qc = QuantumCircuit(qr, cr)
        qc.prepare_state(initial_state, qr)
        qc.barrier()
        
        circuit_groups = []
        circuits = []
        for time in times:
            evo = PauliEvolutionGate(op, time=time, synthesis=synthesis)
            qc_evolution = qc.copy()
            qc_evolution.append(evo.definition, qr)
            qc_evolution.barrier()
            
            for observable in self.observables:
                qc_measurement = qc_evolution.copy()
                for i, obs in enumerate(observable):
                    qc_measurement.append(self.meas_circuits[obs], [i])
                qc_measurement.barrier()
                qc_measurement.measure(qr, cr)
                circuits.append(qc_measurement)
                
                if len(circuits) == batch_size:
                    circuit_groups.append(circuits)
                    circuits = []
        if len(circuits) > 0:
            circuit_groups.append(circuits)
        return circuit_groups
    
    def measurement_circuits(self):
        meas_z = QuantumCircuit(1, name="PauliMeasZ")
        meas_x = QuantumCircuit(1, name="PauliMeasX")
        meas_y = QuantumCircuit(1, name="PauliMeasY")
        (meas_x.h(0), meas_y.sdg(0), meas_y.h(0))
        return {"Z" : meas_z, "X" : meas_x, "Y" : meas_y}
