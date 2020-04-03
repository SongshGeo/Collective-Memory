#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/19 19:08
"""

# import from public libraries
from matplotlib import pyplot as plt
import math
import numpy as np
import pandas as pd

# import from this project
from data_process import get_population, get_actual_water_series, W_MAX, W_MIN
from collective_memory import main_function
from fit_parameters import get_last_k
from plot import plot_initial_sets


d_max, d_mean, d_min = get_population()  # Actual population data.
err = d_max - d_mean  # error bar between passive and positive estimation.

TEST_START_YEAR, TEST_END_YEAR = (1940, 2019)
LEVEE_YEAR_1 = 1983
t_arr = d_mean.loc[TEST_START_YEAR: TEST_END_YEAR].index


def goodness_of_fit(yo, ym):
    sst = ((yo - yo.mean()) ** 2).sum()
    sse = ((yo - ym) ** 2).sum()
    return 1 - sse / sst


def adjusted_goodness(yo, ym):
    n = len(yo)
    sst = ((yo - yo.mean()) ** 2).sum()
    sse = ((yo - ym) ** 2).sum()
    df_e = n - 2
    df_t = n - 1
    return 1 - (sse / df_e) / (sst / df_t)


def fit_nash(yo, ym):
    a = ((yo - ym) ** 2).sum()
    b = ((yo - yo.mean()) ** 2).sum()
    return 1 - a / b


def get_models_results(pop_growth='exp'):

    def exp(t, t_0, n0, k):
        e = math.e
        return n0 * e ** (k * (t - t_0))

    def second_model(t_1, t_2, n1, n2, ka, kb):
        e = math.e
        t01, t02 = t_1[0], t_1[-1]
        p1 = n1 * e ** (ka * (t_1 - t01))
        p2 = n2 * e ** (kb * (t_2 - t02))
        p = np.array(list(p1) + list(p2))
        return p

    # parameters
    t0 = TEST_START_YEAR
    t1 = LEVEE_YEAR_1
    k1, k21, k22, k_osm, k_iudm = [get_last_k()[key] for key in ['k1', 'k21', 'k22', 'osm', 'iudm']]
    ser = get_actual_water_series(TEST_START_YEAR, TEST_END_YEAR)
    simu = main_function(ser, 1, pop_growth=pop_growth)

    # Do simulating
    first_model = exp(t_arr, t0, d_mean[t0], k1)
    second_model = second_model(t_arr[t_arr <= t1], t_arr[t_arr > t1], d_mean[t0], d_mean[t1], ka=k21, kb=k22)
    iudm_model = simu['d_iudm'].loc[t_arr]
    osm_model = simu['d_osm'].loc[t_arr]

    # results
    simulations = {'first': first_model, 'second': second_model, 'IUDM': iudm_model, 'OSM': osm_model}
    return simulations


def different_models_test(ax, pop_growth='exp'):
    simulation_results = get_models_results(pop_growth)
    y_data = d_mean.loc[TEST_START_YEAR:TEST_END_YEAR]
    goodness = pd.DataFrame(index=list(simulation_results.keys()))

    for key in simulation_results:
        y_model = simulation_results[key]
        r2 = goodness_of_fit(y_data, y_model)
        nash = fit_nash(y_data, y_model)
        goodness.loc[key, 'nash'] = nash
        goodness.loc[key, 'r^2'] = r2
        ax.plot(t_arr, y_model, label=r"The {} model".format(key) + r"(\textsl{E} = " + "{:.2f})".format(nash))
    ax.errorbar(d_mean.index, d_mean, fmt="bo:", yerr=err.values, label="Data with error bar")

    # beautification
    ax.set_xlim(TEST_START_YEAR, TEST_END_YEAR)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Years")
    ax.set_ylabel("Relative population density")
    ax.legend(loc=2)
    return goodness


def plot_fig4(pop_growth='exp'):
    fig, ax = plt.subplots(figsize=(12, 5))
    goodness = different_models_test(ax, pop_growth=pop_growth)
    plt.show()
    return goodness


def plot_fig5(pop_growth='exp'):
    ser = get_actual_water_series(TEST_START_YEAR, TEST_END_YEAR)
    df = main_function(ser, steps=1, pop_growth=pop_growth)

    y, w, h = df['y'], df['w'], df['h']
    u, v, m = df['u'], df['v'], df['m']
    u_add_v = df['u'] + df['v']
    d_osm, d_iudm = df['d_osm'], df['d_iudm']

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(311)
    ax1.bar(y, w, color="b", label=r"High water level (\textsl{W})")
    ax1.plot(y, h, "k", label=r"Height of levee (\textsl{H})")
    ax1.set_ylim(W_MIN, W_MAX)
    ax1.set_xlim(y.min() - 1, y.max())
    ax1.set_ylabel("( m )")
    plt.legend(loc=1)
    plt.legend(loc=1)

    ax4 = fig.add_subplot(323)
    ax4.plot(y, m, "-", label=r"Collective memory (\textsl{M})")
    ax4.set_xlim(y.min(), y.max())
    ax4.set_ylim(0, 0.05)
    plt.legend(loc=1)

    ax3 = fig.add_subplot(325)
    ax3.plot(y, d_osm, "r", label=r"Simulated relative population density (\textsl{Dm})")
    ax3.set_xlim(y.min(), y.max())
    ax3.set_xlabel("Year")

    ax2 = fig.add_subplot(324)
    ax2.plot(y, u_add_v, "-", label=r"Collective memory (\textsl{M})")
    ax2.plot(y, u, "--", color="c", label=r"Communicative memory (\textsl{U})")
    ax2.plot(y, v, "--", color="m", label=r"Cultural memory (\textsl{V})")
    ax2.set_xlim(y.min(), y.max())
    ax2.set_ylim(0, 0.05)

    ax5 = fig.add_subplot(326)
    ax5.plot(y, d_iudm, "r", label=r"Simulated relative population density (\textsl{Dm})")
    ax5.set_xlabel("Year")
    for ax in [ax3, ax5]:
        ax.errorbar(d_mean.index, d_mean, fmt="bo:", yerr=err.values,
                    label=r"Observed relative population density (\textsl{Do})")
        ax.set_xlim(y.min(), y.max())
        ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_initial_sets()
    test_results = plot_fig4()
    plot_fig5()
    pass
