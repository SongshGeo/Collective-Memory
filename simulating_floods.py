#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/19 11:24
"""

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

""" Global Parameters """
THETA = 0.19  # Shape parameter of the distribution in the forcing function


def cdf_water_level(theta, w):
    return 1 - (1 - (theta * w / (1 + theta)) ** (1 / theta))


def inverse_cdf_water_level(theta, w):
    return ((1 + theta) / theta) * (1 - (1 - w) ** theta)


def random_water_level(theta):
    u = random.uniform(0., 1.)  # Evenly distributed between 0~1
    w = inverse_cdf_water_level(theta, u)
    return w


def generate_random_flood_series(years, random_state=None, fre=0):
    """
    To generate a series of flood with maximum of water level, randomly.
    :param fre: Increasing possibility of flooding.
    :param years: The length of flood series (year).
    :param random_state: Random seedling set
    :return: A series, with year as index, water level as value.
    """
    random.seed(random_state)  # random seed.
    ser = pd.Series(np.zeros(years), index=np.arange(years))  # To store results
    t = 0
    for flag in np.arange(years):
        pt = 1 - np.e ** (-t)  # Probability of flooding.
        ran = random.uniform(fre, 1.0)
        if ran < pt:    # flooding
            w = random_water_level(THETA)  # Water level
            t = 0  # Initiate the possibility of flooding
            ser[flag] = w  # The water level in the "flag" year.
        if ran >= pt:   # no flooding
            t += 1
        flag += 1
    return ser


def show_one_of_flood_ser(years):
    # Rectify font
    from matplotlib import rc
    rc('text', usetex=True)
    rc('text.latex', preamble=r"""
    \usepackage[eulergreek]{sansmath}\sansmath
    \renewcommand{\rmdefault}{phv} % Arial
    \renewcommand{\sfdefault}{phv} % Arial
    """)

    ser = generate_random_flood_series(years)
    plt.bar(ser.index, ser.values)
    plt.show()


if __name__ == '__main__':
    show_one_of_flood_ser(100)
