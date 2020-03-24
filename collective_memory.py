#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/18 22:41
"""

from simulating_floods import generate_random_flood_series
from data_process import actual_population, get_last_k, get_last_pqr, get_actual_levee_height

import pandas as pd
import numpy as np


""" Initial Values """
P_MIN: float = 160 * 10000 * 2.5  # Pessitive estimation of the population limits
P_MAX: float = 200 * 10000 * 2.5  # positive estimation of the population limits
THRESHOLD_F = 0  # The threshold for enhance.
START_YEAR = 1940  # The start year of this study.
INITIAL_POP = actual_population[START_YEAR]  # Population in the start year (1940).
P, R, Q = [get_last_pqr()[k] for k in ['p', 'r', 'q']]  # Fitted parameters of integrate model
K_OSM, K_IUDM = [get_last_k()[k] for k in ['osm', 'iudm']]  # Use the last result of optimise

""" Global Parameters """

# See Methods: Parameter estimations, Line 183 -- 188
XI_H, KAPPA_T, EPSILON_T = 0, 0, 0  # we ignore the techonology module

# Estimated from literatures, since this water level would affect 952.08 km2 of the study area,
# which is 14.43% of the total area (F = 0.1443), according to (Yuan and Tian 2016).
ALPHA_H = 84.5  # Relationship between ﬂoodwater levels and relative damage

# Since the highest score is 10 (referring to α_D=1), we used the average of these three scores to simulate:
# Reliability of levees (score = 7.31),
# Flood prediction (score = 7.35),
# Emergency response (score = 7.54).
ALPHA_D = 0.75  # Public’s perception of risk

# Because the maximum relative population growth rate ρ_D is difficult to observe, either directly or indirectly,
# we estimated ρ_D by a least-squares method.
RHO_D = 0.00935  # estimated by fitting

# According to (Di Baldassarre et al. 2017)
ETA_H = 1.1  # Safety factor for raising levees

# According to (Di Baldassarre et al. 2017)
MIU_S = 0.04  # Decay rate of collective memory, from the osm


def get_initial_values():
    """
    Generate inital values to input the main model.
    :return: initial values.
    """
    h = 1135.3  # Height of levee in the start year
    d = 2 * INITIAL_POP / (P_MAX + P_MIN)
    y = 0
    w, u, v, m, f = 0, 0, 0, 0, 0
    d_osm = d
    d_iudm = d
    initial_values = [y, w, u, v, m, d_osm, d_iudm, h, f]
    return initial_values


def osm(state, steps, k_osm):
    f, m, d = state
    delta_m = - MIU_S * m + f * d
    delat_d = d * (k_osm - ALPHA_D * m)
    new_m = m + delta_m * steps
    new_d = d + delat_d * steps
    return [new_m, new_d]


def iudm(state, steps, k_iudm):
    f, u, v, d = state
    m = u + v
    delta_u = -(P + R) * u + f * d
    delta_v = -Q * v + R * u
    delta_d = d * (k_iudm - ALPHA_D * m)
    new_u = u + delta_u * steps
    new_v = v + delta_v * steps
    new_d = d + delta_d * steps
    return [new_u, new_v, new_d]


def get_levee_height(state, how):
    y, w, u, v, m, d_osm, d_iudm, h, f = state
    if how == 'enhance':  # If we can heighten levee after floodings.
        if f >= THRESHOLD_F:
            r = ETA_H * (w + XI_H - h)
            h = r + h
        else:
            pass
    elif how == 'fixed':
        h = get_actual_levee_height(y)
    return h
        

def judge_damage(w, h):
    if w > h:
        f = 1 - np.e ** (-(XI_H + w - h) / ALPHA_H)
    else:
        f = 0
    return f


def main_function(flood_ser, steps, how='fixed', k_osm=K_OSM, k_iudm=K_IUDM):
    y, w, u, v, m, d_osm, d_iudm, h, f = get_initial_values()
    df = pd.DataFrame(index=flood_ser.index, columns=['y', 'w', 'u', 'v', 'm', 'd_osm', 'd_iudm', 'h', 'f'])
    for y in flood_ser.index:
        w = flood_ser[y]
        f = judge_damage(w, h)
        state_iudm = [f, u, v, d_iudm]
        state_osm = [f, m, d_osm]
        u, v, d_iudm = iudm(state_iudm, steps, k_iudm)
        m, d_osm = osm(state_osm, steps, k_osm)
        state_all = np.array([y, w, u, v, m, d_osm, d_iudm, h, f])
        h = get_levee_height(state_all, how)  # Levee height in the next year
        df.loc[y, :] = state_all
    return df


if __name__ == '__main__':
    ser = generate_random_flood_series(100)
    result = main_function(ser, 1, how='fixed')
    pass
