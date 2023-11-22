from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeSherbrooke, FakePerth, FakeLagosV2, FakeNairobiV2, FakeGuadalupeV2

FAKE_PROVIDERS = {
    'perth': FakePerth(),
    'nairobi': FakeNairobiV2(),
    'guadalupe': FakeGuadalupeV2(),
    'lagos': FakeLagosV2(),
    'sherbrooke': FakeSherbrooke()
    }

class BackendService:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BackendService, cls).__new__(cls, *args, **kwargs)
            cls.service = QiskitRuntimeService() 
            cls.backends = {backend.name: backend for backend in cls.service.backends()}
        return cls._instance

class BaseBackend:
    def __init__(self, **kwargs):
        self.sampler = self.init_sampler(**kwargs)

    def init_options(self, backend, fake, method, seed, shots, 
                     optimization, resilience, max_parallel_experiments):
        self.service = BackendService().service if backend else None
        self.backend = BackendService().backends.get(backend, None) if backend else None
        self.fake_backend = FAKE_PROVIDERS.get(fake, None)
        self.simulator_options = {
            "seed_simulator": seed,
            "method": method if method else 'statevector',
            "coupling_map": self.fake_backend.coupling_map if self.fake_backend else None,
            "noise_model": NoiseModel.from_backend(self.fake_backend) if self.fake_backend else None,
            "max_parallel_experiments": max_parallel_experiments
            }
        self.transpile_options = {"seed_transpiler": seed}
        self.run_options = {"seed": seed, "shots": shots}
        
        return Options(optimization_level=optimization, resilience_level=resilience, 
                        transpilation=self.transpile_options, execution=self.run_options,
                        simulator=self.simulator_options,)

class CloudBackend(BaseBackend):
    def __init__(self, backend=None, fake=None, method=None,
                 max_parallel_experiments=0, seed=0, shots=10000, optimization=3, resilience=2):
        self.options = self.init_options(backend, fake, method, seed, shots, 
                                         optimization, resilience, max_parallel_experiments)
        self.session = Session(service=self.service, backend=self.backend)
        self.sampler = Sampler(session=self.session, options=self.options)
    
class LocalBackend(BaseBackend):
    def __init__(self, fake=None, method=None,
                 max_parallel_experiments=0, seed=0, shots=10000, optimization=3, resilience=2):
        backend = None
        self.options = self.init_options(backend, fake, method, seed, shots, 
                                         optimization, resilience, max_parallel_experiments)
        self.sampler = AerSampler(backend_options=self.options.simulator.__dict__,
                                  transpile_options={"seed_transpiler": seed},
                                  run_options=self.options.execution.__dict__)

def get_fake_backends():
    return list(FAKE_PROVIDERS.keys())

def get_cloud_backends():
    return BackendService().backends.keys()

def save_credentials(token):
    QiskitRuntimeService.save_account(token, overwrite=True)
