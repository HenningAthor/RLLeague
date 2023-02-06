"""
File to hold utility functions.
"""
from typing import Union

import numpy as np


def scale_continuous(x: Union[float, np.ndarray],
                     x_min: float,
                     x_max: float,
                     lower: float,
                     upper: float) -> Union[float, np.ndarray]:
    """
    Transforms inv_w in to the interval [lower, upper].

    :return: Transformed inv_w
    """
    # transform into new interval
    return ((x - x_min) / (x_max - x_min)) * (upper - lower) + lower


def scale_discrete(x: Union[float, np.ndarray],
                   x_min: float,
                   x_max: float,
                   lower: float,
                   upper: float) -> Union[float, np.ndarray]:
    """
    Transforms x in to the interval [lower, upper]. Then discretize x to the
    nearest integer.

    :return: Transformed x
    """
    # transform into new interval
    x = ((x - x_min) / (x_max - x_min)) * (upper - lower) + lower
    return np.rint(x)


random_numbers = {0: np.random.sample(size=10000).tolist()}


def random_sample():
    """
    Function to quickly generate a new sample between [0, 1].

    :return: Sample between [0, 1]
    """
    if not random_numbers[0]:
        random_numbers[0] = np.random.sample(size=10000).tolist()

    return random_numbers[0].pop()
