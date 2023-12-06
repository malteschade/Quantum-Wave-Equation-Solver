#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}

{
    MIT License

    Copyright (c) [2023] [Malte Leander Schade]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Dict, List

# Other modules
import numpy as np

# -------- CONSTANTS --------
FORWARD_FD_COEFF: Dict[int, List[float]] = {
        1: [-1, 1],
        2: [-3/2, 2, -1/2],
        3: [-11/6, 3, -3/2, 1/3],
        4: [-25/12, 4, -3, 4/3, -1/4]
    }

# -------- CLASSES --------
class FDTransform1DA:
    def __init__(self, mu: np.ndarray, rho: np.ndarray, dx: float, nx: int,
                 order: int, bcs: dict) -> None:
        self.mu = mu
        self.rho = rho
        self.dx = dx
        self.nx = nx
        self.order = order
        self.bcs = bcs

        # Define FD operator
        self.d = boundary(scale(self.get_d(self.order, self.nx, self.dx), rows=1), self.bcs)

        # Define mass matrices
        self.sqrt_m = self.get_sqrt_m(self.rho, self.get_z(self.nx))
        self.inv_sqrt_m = self.get_inv_sqrt_m(self.rho, self.get_z(self.nx))

        # Define cholesky decomposition
        self.u = self.get_u(self.rho, self.mu, self.d)

        # Define stiffness matrix
        self.k = self.get_k(self.u)

        # Define impedance matrix
        self.q = self.get_q(self.k, self.get_z(self.nx), self.get_i(self.nx))

        # Define transformation matrices
        self.t = self.get_t(self.u, scale(self.get_z(self.nx), rows=1),
                        scale(self.get_i(self.nx), rows=1))
        self.inv_t = self.get_inv_t(self.t)

        # Define hamiltonian
        self.h = self.get_h(scale(self.u, cols=1), self.get_z(self.nx+1))

    def get_d(self, order: int, length: int, dx: float) -> np.ndarray:
        return (1/dx) * (1/order) * np.sum([np.diag(np.full(length-k, c), k=-k)
                       for k, c in enumerate(FORWARD_FD_COEFF[order])], axis=0)

    def get_z(self, length: int) -> np.ndarray:
        return np.zeros((length, length))

    def get_i(self, length: int) -> np.ndarray:
        return np.identity(length)

    def get_sqrt_m(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.block([[np.diag(np.sqrt(rho)), z],
                         [z, np.diag(np.sqrt(rho))]])

    def get_inv_sqrt_m(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.block([[np.diag(np.sqrt(1/rho)), z],
                         [z, np.diag(np.sqrt(1/rho))]])

    def get_u(self, rho: np.ndarray, mu: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.diag(np.sqrt(mu)) @ d @ np.diag(np.sqrt(1/rho))

    def get_k(self, u: np.ndarray)  -> np.ndarray:
        return -u.T @ u

    def get_q(self, k: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        return np.block([[z, i],[k, z]])

    def get_t(self, u: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        return np.block([[u, z],[z, i]])

    def get_inv_t(self, t: np.ndarray) -> np.ndarray:
        return np.linalg.inv(t.T @ t) @ t.T # Left inverse | Least squares (full rank)

    def get_h(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.block([[z, 1j*u],[-1j*u.T, z]])

    def get_dict(self) -> dict:
        return {'h': self.h,
                't': self.t,
                'inv_t': self.inv_t,
                'k': self.k,
                'q': self.q,
                'u': self.u,
                'd': self.d,
                'sqrt_m': self.sqrt_m,
                'inv_sqrt_m': self.inv_sqrt_m}

# -------- FUNCTIONS --------
def scale(array: np.ndarray, rows: int = 0, cols: int = 0) -> np.ndarray:
    l = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    l[:array.shape[0], :array.shape[1]] = array
    return l

def boundary(array: np.ndarray, bcs: dict) -> np.ndarray:
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if bcs[side] == 'DBC':
            array[index, index] = 1
        elif bcs[side] == 'NBC':
            array[index, index] = 0
        else:
            pass
    return array
