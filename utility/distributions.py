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

# -------- FUNCTIONS --------
def _check_inputs(**kwargs) -> None:
    assert kwargs.get('length', 1) > 0, 'length must be positive'
    assert kwargs.get('position', 1) >= 0, 'position must be non-negative'
    assert kwargs.get('position', 0) < kwargs.get('length', 1), 'position must be less than length'
    assert kwargs.get('sigma', 1) > 0, 'sigma must be positive'
    assert kwargs.get('power', 1) > 0, 'power must be positive'

def spike(value: float, length: int, position: int) -> np.ndarray:
    _check_inputs(value=value, length=length, position=position)
    return np.concatenate([np.zeros(position), np.array([value]), np.zeros(length-position-1)])

def homogeneous(value: float, length: int) -> np.ndarray:
    _check_inputs(value=value, length=length)
    return np.ones(length) * value

def linear(value_start: float, value_end: float, length: int) -> np.ndarray:
    _check_inputs(value_start=value_start, value_end=value_end, length=length)
    return np.linspace(value_start, value_end, length)

def polynomial(value_start: float, value_end: float, length: int, power: float = 2) -> np.ndarray:
    _check_inputs(value_start=value_start, value_end=value_end, length=length, power=power)
    return np.linspace(value_start**power, value_end**power, length)**(1/power)

def exponential(value_start: float, value_end: float, length: int) -> np.ndarray:
    _check_inputs(value_start=value_start, value_end=value_end, length=length)
    return np.exp(np.linspace(np.log(value_start), np.log(value_end), length))

def gaussian(value: float, length: int, position: int, sigma: float = 1, offset: float = 0) -> np.ndarray:
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    return value * np.exp(-(np.arange(length) - position)**2 / (2 * sigma**2)) + offset

def raised_cosine(value: float, length: int, position: int, sigma: float = 1, offset: float = 0) -> np.ndarray:
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    x = (np.arange(length) - position) / sigma
    return value * 0.5 * (1 + np.cos(np.pi * x)) * np.where(np.abs(x) < 1, 1, 0) + offset

def ricker(value: float, length: int, position: int, sigma: float = 1, offset: float = 0) -> np.ndarray:
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    x = (np.arange(length) - position) / sigma
    return value * (1 - 2 * np.pi**2 * x**2) * np.exp(-np.pi**2 * x**2) + offset

def sinc(value: float, length: int, position: int, sigma: float = 1, offset: float = 0) -> np.ndarray:
    _check_inputs(value=value, length=length, position=position)
    x = np.arange(length) - position
    return np.sinc(x / sigma) * value + offset
