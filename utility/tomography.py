import numpy as np
from qiskit_experiments.library.tomography.basis import PauliMeasurementBasis
from qiskit_experiments.library.tomography.fitters import (linear_inversion, 
                                                           scipy_linear_lstsq, scipy_gaussian_lstsq,
                                                           cvxpy_linear_lstsq, cvxpy_gaussian_lstsq)

class TomographyReal:
    def __init__(self, logger, fitter='linear'):
        self.logger = logger
        self.fitter = fitter

    def run_tomography(self, result_groups, observables, times):
        measurements = [result for result_group in result_groups for result in result_group.decompose()]
        quasi_dist = [m.quasi_dists[0] for m in measurements]
        max_key = max(max(dist.keys()) for dist in quasi_dist)
        [dist.setdefault(i, 0) for dist in quasi_dist for i in range(max_key + 1)]
        freq = np.array([dist[key] for dist in quasi_dist for key in range(max_key + 1)])
        freq = np.reshape(freq, (len(quasi_dist), max_key+1))
        
        shot_data = np.array([meta['shots'] for meta in measurements[0].metadata]*len(observables))
        measurement_data = np.array([[{'Z': 0, 'X': 1, 'Y': 2}[char] for char in sublist] for sublist in observables])
        preparation_data = np.full((len(shot_data), 0), None)
        
        fitter_kwargs = {
            "measurement_basis": PauliMeasurementBasis(),
            "measurement_qubits": tuple(range(len(measurement_data[0])))
        }

        states = []
        for i in range(len(times)):
            self.logger.info(f'Tomography step: {i+1} | {len(times)}.')
            freq_step = freq[i*len(observables):(i+1)*len(observables)]
            outcome_data = np.expand_dims(freq_step * shot_data[:, np.newaxis], axis=0)
            match self.fitter:
                case 'linear':
                    rho, metadata = linear_inversion(outcome_data, shot_data,measurement_data,
                                                    preparation_data, **fitter_kwargs)
                case 'cvxpy_gaussian':
                    rho, metadata = cvxpy_gaussian_lstsq(outcome_data, shot_data, measurement_data,
                                                        preparation_data, **fitter_kwargs)
                case 'cvxpy_linear':
                    rho, metadata = cvxpy_linear_lstsq(outcome_data, shot_data, measurement_data,
                                                    preparation_data, **fitter_kwargs)
                case 'scipy_gaussian':
                    rho, metadata = scipy_gaussian_lstsq(outcome_data, shot_data, measurement_data,
                                                        preparation_data, **fitter_kwargs)
                case 'scipy_linear':
                    rho, metadata = scipy_linear_lstsq(outcome_data, shot_data, measurement_data,
                                                    preparation_data, **fitter_kwargs)
            eigval, eigvec = np.linalg.eig(rho)
            self.logger.debug(f'Eigenvalues: {eigval}')
            self.logger.debug(f'Eigenvectors: {eigvec}')
            states.append(eigvec[:, np.argmax(eigval)])   
        return np.array(states)

def parallel_transport(states, initial_state):
    corrected_states = [initial_state]
    for state in states:
        ref_state = corrected_states[-1]
        phase = np.imag(np.log(state.dot(np.conj(ref_state))))
        corrected_states.append(np.exp(-1j * phase) * state)
    return np.array(corrected_states)
    