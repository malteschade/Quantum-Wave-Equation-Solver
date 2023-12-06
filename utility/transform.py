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
    """
    Class that calculates the transformation matrices and hamiltonian
    for the 1D elastic wave equation solver subject to boundary conditions.
    """

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

    def get_z(self, length: int) -> np.ndarray:
        """
        Returns a zero matrix with specified size.

        Args:
            length (int): The size of the matrix.


        Returns:
            np.ndarray: The zero matrix.
        """
        return np.zeros((length, length))

    def get_i(self, length: int) -> np.ndarray:
        """
        Returns an identity matrix with specified size.

        Args:
            length (int): The size of the matrix.


        Returns:
            np.ndarray: The identity matrix.
        """
        return np.identity(length)

    def get_sqrt_m(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the square root matrix of the medium densities.

        Args:
            rho (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.

        Returns:
            np.ndarray: The square root mass matrix.
        """
        return np.block([[np.diag(np.sqrt(rho)), z],
                         [z, np.diag(np.sqrt(rho))]])

    def get_inv_sqrt_m(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse square root matrix of the medium densities.
        
        Args:
            rho (np.ndarray): The medium densities.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The inverse square root mass matrix.
        """
        return np.block([[np.diag(np.sqrt(1/rho)), z],
                         [z, np.diag(np.sqrt(1/rho))]])

    def get_d(self, order: int, length: int, dx: float) -> np.ndarray:
        """
        Calculates the 1D forward Finite-Difference (FD) matrix of n-th order.

        Args:
            order (int): The order of the FD matrix.
            length (int): The length of the FD matrix.
            dx (float): The grid spacing.

        Returns:
            np.ndarray: The 1D FD matrix.
        """
        return (1/dx) * (1/order) * np.sum([np.diag(np.full(length-k, c), k=-k)
                       for k, c in enumerate(FORWARD_FD_COEFF[order])], axis=0)

    def get_u(self, rho: np.ndarray, mu: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Calculates the analytical Cholesky decomposition of
        the FD operator with the medium parameters.
        
        Args:
            rho (np.ndarray): The medium densities.
            mu (np.ndarray): The medium viscosities.
            d (np.ndarray): The 1D FD matrix.
        
        Returns:
            np.ndarray: The analytic Cholesky decomposition matrix.
        """

        return np.diag(np.sqrt(mu)) @ d @ np.diag(np.sqrt(1/rho))

    def get_k(self, u: np.ndarray)  -> np.ndarray:
        """
        Calculates the stiffness matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            
        Returns:
            np.ndarray: The stiffness matrix.
        """
        return -u.T @ u

    def get_q(self, k: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        """
        Calculates the impedance matrix.
        
        Args:
            k (np.ndarray): The stiffness matrix.
            z (np.ndarray): The zero matrix.
            i (np.ndarray): The identity matrix.
        
        Returns:
            np.ndarray: The impedance matrix.
        """
        return np.block([[z, i],[k, z]])

    def get_t(self, u: np.ndarray, z: np.ndarray, i: np.ndarray) -> np.ndarray:
        """
        Calculates the transformation matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            i (np.ndarray): The identity matrix.
            
        Returns:
            np.ndarray: The transformation matrix.
        """
        return np.block([[u, z],[z, i]])

    def get_inv_t(self, t: np.ndarray) -> np.ndarray:
        """
        Calculates the inverse transformation matrix with 
        least-squares (left inverse).
        
        Args:
            t (np.ndarray): The transformation matrix.
            
        Returns:
            np.ndarray: The inverse transformation matrix.
        """
        return np.linalg.inv(t.T @ t) @ t.T # Left inverse | Least squares (full rank)

    def get_h(self, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates the hamiltonian matrix.
        
        Args:
            u (np.ndarray): The Cholesky decomposition matrix.
            z (np.ndarray): The zero matrix.
            
        Returns:
            np.ndarray: The hamiltonian matrix.
        """
        return np.block([[z, 1j*u],[-1j*u.T, z]])

    def get_dict(self) -> dict:
        """
        Returns a dictionary containing the transformation matrices.
        
        Returns:
            dict: The transformation matrices.
        """
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
    """
    Scales a matrix by adding rows and columns of zeros.
    
    Args:
        array (np.ndarray): The matrix to be scaled.
        rows (int, optional): The number of rows to be added. Defaults to 0.
        cols (int, optional): The number of columns to be added. Defaults to 0.
    
    Returns:
        np.ndarray: The scaled matrix.
    """
    l = np.zeros((array.shape[0]+rows, array.shape[1]+cols))
    l[:array.shape[0], :array.shape[1]] = array
    return l

def boundary(array: np.ndarray, bcs: dict) -> np.ndarray:
    """
    Applies boundary conditions to a matrix.
    
    Args:
        array (np.ndarray): The matrix to be modified.
        bcs (dict): The boundary conditions.
        
    Returns:
        np.ndarray: The modified matrix.
    """
    for side in ['left', 'right']:
        index = 0 if side == 'left' else -1
        if bcs[side] == 'DBC':
            array[index, index] = 1
        elif bcs[side] == 'NBC':
            array[index, index] = 0
        else:
            pass
    return array
