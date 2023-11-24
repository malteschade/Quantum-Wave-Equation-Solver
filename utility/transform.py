import numpy as np

FORWARD_FD_COEFF = {
        1: [-1, 1],
        2: [-3/2, 2, -1/2],
        3: [-11/6, 3, -3/2, 1/3],
        4: [-25/12, 4, -3, 4/3, -1/4]
    }

def scale(array: np.ndarray, rows: int = 0, cols: int = 0) -> np.ndarray:
    L = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    L[:array.shape[0], :array.shape[1]] = array
    return L

def boundary(array: np.ndarray, boundary: dict) -> np.ndarray:
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if boundary[side] == 'DBC':
            array[index, index] = 1
        elif boundary[side] == 'NBC':
            array[index, index] = 0
        else:
            pass
    return array

class FDTransform1DA:
    def __init__(self, mu, rho, dx, nx, order, bcs):
        self.mu = np.array(mu)
        self.rho = np.array(rho)
        self.dx = dx
        self.nx = nx
        self.order = order
        self.bcs = bcs
        
        # Define FD operator
        self.d = boundary(scale(self.D(self.order, self.nx, self.dx), rows=1), self.bcs)
        
        # Define mass matrices
        self.M(self.rho, self.Z(self.nx))
        
        # Define cholesky decomposition
        self.u = self.U(self.rho, self.mu, self.d)

        # Define stiffness matrix
        self.k = self.K(self.u)

        # Define impedance matrix
        self.q = self.Q(self.k, self.Z(self.nx), self.I(self.nx))

        # Define transformation matrices
        self.t = self.T(self.u, scale(self.Z(self.nx), rows=1), scale(self.I(self.nx), rows=1))
        self.inv_t = self.INV_T(self.t)

        # Define hamiltonian
        self.h = self.H(scale(self.u, cols=1), self.Z(self.nx+1))
    
    def D(self, order: int, length: int, dx: float) -> np.ndarray:
        return (1/dx) * (1/order) * np.sum([np.diag(np.full(length-k, c), k=-k)
                       for k, c in enumerate(FORWARD_FD_COEFF[order])], axis=0)
        
    def Z(self, length: int) -> np.ndarray:
        return np.zeros((length, length))
    
    def I(self, length: int) -> np.ndarray:
        return np.identity(length)
    
    def SQRT(self, param: np.ndarray) -> np.ndarray:
        return np.diag(np.sqrt(param))
    
    def M(self, rho: np.array, Z: np.array) -> np.ndarray:
        self.sqrt_m = np.block([[self.SQRT(rho), Z],
                                [Z, self.SQRT(rho)]])
        self.inv_sqrt_m = np.block([[self.SQRT(1/rho), Z],
                                    [Z, self.SQRT(1/rho)]]) 
    
    def U(self, rho: np.ndarray, mu: np.ndarray, D: np.ndarray) -> np.ndarray:
        return self.SQRT(mu) @ D @ self.SQRT(1/rho)
    
    def K(self, U: np.ndarray)  -> np.ndarray:
        return -U.T @ U
    
    def Q(self, K: np.ndarray, Z: np.ndarray, I: np.ndarray) -> np.ndarray:
        return np.block([[Z, I],[K, Z]])
    
    def T(self, U: np.ndarray, Z: np.ndarray, I: np.ndarray) -> np.ndarray:
        return np.block([[U, Z],[Z, I]])
    
    def INV_T(self, T: np.ndarray) -> np.ndarray:
        return np.linalg.inv(T.T @ T) @ T.T # Left inverse | Least squares (full rank)
    
    def H(self, U: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.block([[Z, 1j*U],[-1j*U.T, Z]])

    def get_dict(self):
        return {'h': np.imag(self.h).tolist(),
                't': self.t.tolist(),
                'inv_t': self.inv_t.tolist(),
                'k': self.k.tolist(),
                'q': self.q.tolist(),
                'u': self.u.tolist(),
                'd': self.d.tolist(),
                'sqrt_m': self.sqrt_m.tolist(),
                'inv_sqrt_m': self.inv_sqrt_m.tolist()}
