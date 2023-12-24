#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Sub-module for giving different parameter distributions for
initial conditions and material properties.}

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

# -------- FUNCTIONS --------
def _check_inputs(**kwargs) -> None:
    assert kwargs.get('length', 1) > 0, 'length must be positive'
    assert kwargs.get('position', 1) >= 0, 'position must be non-negative'
    assert kwargs.get('position', 0) < kwargs.get('length', 1), 'position must be less than length'
    assert kwargs.get('sigma', 1) > 0, 'sigma must be positive'
    assert kwargs.get('power', 1) > 0, 'power must be positive'

def spike(value: float, length: int, position: int) -> np.ndarray:
    """
    Returns a spike function with a single non-zero value at a given position.
    
    Args:
        value (float): Value of the spike.
        length (int): Length of the spike.
        position (int): Position of the spike.
    
    Returns:
        np.ndarray: The spike function.
    """
    _check_inputs(value=value, length=length, position=position)
    return np.concatenate([np.zeros(position), np.array([value]), np.zeros(length-position-1)])

def homogeneous(value: float, length: int) -> np.ndarray:
    """
    Returns a homogeneous function with the given length.
    
    Args:
        value (float): Value of the homogeneous function.
        length (int): Length of the homogeneous function.
        
    Returns:
        np.ndarray: The homogeneous function.
    """
    _check_inputs(value=value, length=length)
    return np.ones(length) * value

def linear(value_start: float, value_end: float, length: int) -> np.ndarray:
    """
    Returns a linear function with the given length.
    
    Args:
        value_start (float): Starting value of the linear function.
        value_end (float): Ending value of the linear function.
        length (int): Length of the linear function.
    
    Returns:
        np.ndarray: The linear function.
    """
    _check_inputs(value_start=value_start, value_end=value_end, length=length)
    return np.linspace(value_start, value_end, length)

def polynomial(value_start: float, value_end: float, length: int, power: float = 2) -> np.ndarray:
    """
    Returns a polynomial function with the given length.
    
    Args:
        value_start (float): Starting value of the polynomial function.
        value_end (float): Ending value of the polynomial function.
        length (int): Length of the polynomial function.
        power (float): Power of the polynomial function.
    
    Returns:
        np.ndarray: The polynomial function.
    """
    _check_inputs(value_start=value_start, value_end=value_end, length=length, power=power)
    return np.linspace(value_start**power, value_end**power, length)**(1/power)

def exponential(value_start: float, value_end: float, length: int) -> np.ndarray:
    """
    Returns an exponential function with the given length.
    
    Args:
        value_start (float): Starting value of the exponential function.
        value_end (float): Ending value of the exponential function.
        length (int): Length of the exponential function.
    
    Returns:
        np.ndarray: The exponential function.
    """
    _check_inputs(value_start=value_start, value_end=value_end, length=length)
    return np.exp(np.linspace(np.log(value_start), np.log(value_end), length))

def gaussian(value: float, length: int, position: int, sigma: float = 1,
             offset: float = 0) -> np.ndarray:
    """
    Returns a gaussian function with the given length.
    
    Args:
        value (float): Value of the gaussian function.
        length (int): Length of the gaussian function.
        position (int): Position of the gaussian function.
        sigma (float): Sigma of the gaussian function.
        offset (float): Offset of the gaussian function.
        
    Returns:
        np.ndarray: The gaussian function.
    """
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    return value * np.exp(-(np.arange(length) - position)**2 / (2 * sigma**2)) + offset

def raised_cosine(value: float, length: int, position: int, sigma: float = 1,
                  offset: float = 0) -> np.ndarray:
    """
    Returns a raised cosine function with the given length.
    
    Args:
        value (float): Value of the raised cosine function.
        length (int): Length of the raised cosine function.
        position (int): Position of the raised cosine function.
        sigma (float): Sigma of the raised cosine function.
        offset (float): Offset of the raised cosine function.
        
    Returns:
        np.ndarray: The raised cosine function.
    """
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    x = (np.arange(length) - position) / sigma
    return value * 0.5 * (1 + np.cos(np.pi * x)) * np.where(np.abs(x) < 1, 1, 0) + offset

def ricker(value: float, length: int, position: int, sigma: float = 1,
           offset: float = 0) -> np.ndarray:
    """
    Returns a ricker function with the given length.
    
    Args:
        value (float): Value of the ricker function.
        length (int): Length of the ricker function.
        position (int): Position of the ricker function.
        sigma (float): Sigma of the ricker function.
        offset (float): Offset of the ricker function.
    
    Returns:
        np.ndarray: The ricker function.
    """
    _check_inputs(value=value, length=length, position=position, sigma=sigma)
    x = (np.arange(length) - position) / sigma
    return value * (1 - 2 * np.pi**2 * x**2) * np.exp(-np.pi**2 * x**2) + offset

def sinc(value: float, length: int, position: int, sigma: float = 1,
         offset: float = 0) -> np.ndarray:
    """
    Returns a sinc function with the given length.
    
    Args:
        value (float): Value of the sinc function.
        length (int): Length of the sinc function.
        position (int): Position of the sinc function.
        sigma (float): Sigma of the sinc function.
        offset (float): Offset of the sinc function.
    
    Returns:
        np.ndarray: The sinc function.
    """
    _check_inputs(value=value, length=length, position=position)
    x = np.arange(length) - position
    return np.sinc(x / sigma) * value + offset
