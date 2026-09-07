from collections.abc import Callable

import numpy as np
from numpy import float32
from numpy.random import Generator
from numpy.typing import NDArray


def rejection(
    rng: Generator, f: Callable, size: int, x_min: float, x_max: float
) -> NDArray[float32]:
    """
    Rejection sampler for arbitrary probability density functions.

    Args:
        f (Callable): A function f(x) -> u that describes the probability distribution (must be valid between x_min and x_max).
        size (int): The size of the required sample.
        x_min (float): The smallest possible value to sample.
        x_max (float): The largest possible value to sample.

    Returns:
        NDArray[float32]: The produced sample according to f.

    """
    res: NDArray[float32] = np.array([], dtype=float32)
    while res.size < size:
        x: NDArray[float32] = x_min + (x_max - x_min) * rng.random(size, dtype=float32)
        u: NDArray[float32] = rng.random(size, dtype=float32)
        res: NDArray[float32] = np.append(res, x[u < f(x)])
    return res[:size]
