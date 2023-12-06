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
