#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for processing the states and medium properties of a quantum experiment.}

{
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
}
"""

# -------- IMPORTS --------
# Other modules
import numpy as np

# -------- CLASSES --------
class ProcessorBase:
    """
    Base class for processors.
    """
    def __init__(self, n_val1: int, n_val2: int, nt: int) -> None:
        self._n_val1 = n_val1
        self._n_val2 = n_val2
        self._nt = nt
        self.values_1 = np.zeros((nt, n_val1))
        self.values_2 = np.zeros((nt, n_val2))

    def _set_val1(self, values: np.ndarray, index: int) -> None:
        self.values_1[index] = values

    def _set_val2(self, values: np.ndarray, index: int) -> None:
        self.values_2[index] = values

    def _get_vals(self, index) -> tuple([np.ndarray, np.ndarray]):
        return self.values_1[index], self.values_2[index]

class MediumProcessor(ProcessorBase):
    """
    Class that processes the medium properties of a quantum experiment.
    """
    def __init__(self, n_mu: int, n_rho: int) -> None:
        super().__init__(n_mu, n_rho, 1) # Time-invariant medium

    def set_mu(self, values: list) -> None:
        """
        Set the elastic modulus values of the medium.
        
        Args:
            values (list): The elastic modulus values of the medium.
        """
        self._set_val1(values, 0)

    def set_rho(self, values: list) -> None:
        """
        Set the density values of the medium.
        
        Args:
            values (list): The density values of the medium.
        """
        self._set_val2(values, 0)

    def get_medium(self) -> tuple([np.ndarray, np.ndarray]):
        """
        Get the elastic modulus and density values of the medium.
        
        Returns:
            tuple([np.ndarray, np.ndarray]): The elastic modulus and density values of the medium.
        """
        return self._get_vals(0)

    def get_dict(self) -> dict:
        """
        Get the elastic modulus and density values of the medium.
        
        Returns:
            dict: The elastic modulus and density values of the medium.
        """
        return {'mu': self.values_1,
                'rho': self.values_2}

class StateProcessor(ProcessorBase):
    """
    Class that processes the states of a quantum experiment.
    """
    def __init__(self, nx: int, nt: int, shift: int = 0) -> None:
        super().__init__(nx, nx, nt) # Time-dependent state
        self.norm = 1
        self.states = np.zeros((nt, (nx+shift)*2))

    def set_u(self, values: list, index: int) -> None:
        """
        Set the position values of the state.
        
        Args:
            values (list): The position values of the state.
            index (int): The time index of the state.
        """
        assert index < self._nt, 'index must be less than nt'
        self._set_val1(values, index)

    def set_v(self, values: list, index: int) -> None:
        """
        Set the velocity values of the state.
        
        Args:
            values (list): The velocity values of the state.
            index (int): The time index of the state.
        """
        assert index < self._nt, 'index must be less than nt'
        self._set_val2(values, index)

    def get_values(self, index) -> tuple([np.ndarray, np.ndarray]):
        """
        Get the position and velocity values of the state.
        
        Args:
            index (int): The time index of the state.
        
        Returns:
            tuple([np.ndarray, np.ndarray]): The position and velocity values of the state.
        """
        return self._get_vals(index)

    def get_state(self, index) -> np.ndarray:
        """
        Get the statevector of the state.
        
        Args:
            index (int): The time index of the state.
            
        Returns:
            np.ndarray: The statevector of the state.
        """
        state = self.states[index]
        assert np.round(np.linalg.norm(state), 10) == 1, 'state is not a valid statevector'
        return state

    def forward_state(self, index: int, transform: np.ndarray) -> None:
        """
        Forward transform the statevector of the state.
        
        Args:
            index (int): The time index of the state.
            transform (np.ndarray): The transformation matrix.
        """
        state = np.concatenate(self._get_vals(index))
        assert len(state) == transform.shape[1], 'length of state must be equal to transform shape'
        transformed_state = transform @ state
        self.norm = np.linalg.norm(transformed_state)
        self.states[index] = transformed_state / self.norm

    def inverse_state(self, index: int, transform: np.ndarray) -> None:
        """
        Inverse transform the statevector of the state.
        
        Args:
            index (int): The time index of the state.
            transform (np.ndarray): The transformation matrix.
        """
        state = self.states[index]
        assert len(state) == transform.shape[1], 'length of state must be equal to transform shape'
        assert np.linalg.matrix_rank(transform) == transform.shape[0],\
            'transform must be full rank (check boundary conditions)'
        state = transform @ (state * self.norm)
        middle = state.shape[0] // 2
        self._set_val1(state[:middle], index)
        self._set_val2(state[middle:], index)

    def get_dict(self) -> dict:
        """
        Get the position, velocity and statevector values of the state.
        
        Returns:
            dict: The position, velocity and statevector values of the state.
        """
        return {'u': self.values_1,
                'v': self.values_2,
                'state': self.states}
