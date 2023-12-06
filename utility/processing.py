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
# Other modules
import numpy as np

# -------- CLASSES --------
class ProcessorBase:
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
    def __init__(self, n_mu: int, n_rho: int) -> None:
        super().__init__(n_mu, n_rho, 1) # Time-invariant medium

    def set_mu(self, values: list) -> None:
        self._set_val1(values, 0)

    def set_rho(self, values: list) -> None:
        self._set_val2(values, 0)

    def get_medium(self) -> tuple([np.ndarray, np.ndarray]):
        return self._get_vals(0)

    def get_dict(self) -> dict:
        return {'mu': self.values_1,
                'rho': self.values_2}

class StateProcessor(ProcessorBase):
    def __init__(self, nx: int, nt: int, shift: int = 0) -> None:
        super().__init__(nx, nx, nt) # Time-dependent state
        self.norm = 1
        self.states = np.zeros((nt, (nx+shift)*2))

    def set_u(self, values: list, index: int) -> None:
        assert index < self._nt, 'index must be less than nt'
        self._set_val1(values, index)

    def set_v(self, values: list, index: int) -> None:
        assert index < self._nt, 'index must be less than nt'
        self._set_val2(values, index)

    def get_values(self, index) -> tuple([np.ndarray, np.ndarray]):
        return self._get_vals(index)

    def get_state(self, index) -> np.ndarray:
        state = self.states[index]
        assert np.round(np.linalg.norm(state), 10) == 1, 'state is not a valid statevector'
        return state

    def forward_state(self, index: int, transform: np.ndarray) -> None:
        state = np.concatenate(self._get_vals(index))
        assert len(state) == transform.shape[1], 'length of state must be equal to transform shape'
        transformed_state = transform @ state
        self.norm = np.linalg.norm(transformed_state)
        self.states[index] = transformed_state / self.norm

    def inverse_state(self, index: int, transform: np.ndarray) -> None:
        state = self.states[index]
        assert len(state) == transform.shape[1], 'length of state must be equal to transform shape'
        assert np.linalg.matrix_rank(transform) == transform.shape[0],\
            'transform must be full rank (check boundary conditions)'
        state = transform @ (state * self.norm)
        middle = state.shape[0] // 2
        self._set_val1(state[:middle], index)
        self._set_val2(state[middle:], index)

    def get_dict(self) -> dict:
        return {'u': self.values_1,
                'v': self.values_2,
                'state': self.states}
