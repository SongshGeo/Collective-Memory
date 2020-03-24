#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/22 23:08
"""
from data_process import actual_population, P_MEAN
from plot import plot_initial_sets
from matplotlib import pyplot as plt
import numpy as np
from lmfit import Parameters, Minimizer, report_fit
from model_test import fit_nash, goodness_of_fit


# 线性模型是不可能的
plot_initial_sets()
population_density = actual_population / P_MEAN


# 人口模式太符合马尔萨斯指数模型了
def exp_curve(initial_year, diagram=False):
    t_arr, y_data = population_density.index, population_density.values

    def exp_model(t, n, k):
        return np.array(n * np.exp(k * (t - initial_year)))

    def residual(params):
        k, n_1904 = params['k'], params['n']
        return y_data - exp_model(t_arr, n_1904, k)

    fit_params = Parameters()
    for param in ['k', 'n']:
        fit_params.add(name=param, value=0.05, min=0)
    minner = Minimizer(residual, fit_params)
    result = minner.minimize()
    report_fit(result)
    k_exp, n_initial = result.params['k'].value, result.params['n'].value
    y_model = exp_model(t_arr, n_initial, k_exp)

    def plot():
        r2 = goodness_of_fit(y_data, y_model)
        nash = fit_nash(y_data, y_model)

        plt.scatter(t_arr, y_data)
        plt.plot(t_arr, y_model, label="Nash is {:.2f}, $r^2$ is {:.2f}".format(nash, r2))
        plt.legend()
        plt.show()

    if diagram:
        plot()
    return k_exp, n_initial


if __name__ == '__main__':
    exp_curve(1904, diagram=True)
    pass
