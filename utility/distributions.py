import numpy as np

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