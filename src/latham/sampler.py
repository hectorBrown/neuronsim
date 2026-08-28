from collections.abc import Callable

import numpy as np
from numpy.core.numeric import float64
from numpy.typing import NDArray


def rejection(f: Callable, size: int, x_min: float, x_max: float) -> NDArray[float64]:
    """
    Rejection sampler for arbitrary probability density functions.

    Args:
        f (Callable): A function f(x) -> u that describes the probability distribution (must be valid between x_min and x_max).
        size (int): The size of the required sample.
        x_min (float): The smallest possible value to sample.
        x_max (float): The largest possible value to sample.

    Returns:
        NDArray[float64]: The produced sample according to f.

    """
    res: NDArray[float64] = np.array([])
    while res.size < size:
        x = np.random.uniform(x_min, x_max, size)
        u = np.random.uniform(0, 1, size)
        res: NDArray[float64] = np.append(res, x[u < f(x)])
    return res[:size]
