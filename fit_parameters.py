#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/19 19:14
"""
# import from external libraries
from lmfit import Minimizer, Parameters, report_fit
from matplotlib import pyplot as plt

# import from this project
from data_process import *
from collective_memory import osm, iudm, judge_damage

d_max, d_mean, d_min = get_population()  # Actual population data.
FIT_START_YEAR, FIT_END_YEAR = (1940, 2018)  # Fit parameters between 1949 and 2019
LEVEE_YEAR_1 = 1983
t_arr = d_mean.loc[FIT_START_YEAR: FIT_END_YEAR].index  # Year index used in fitting
p_arr = d_mean.loc[FIT_START_YEAR: FIT_END_YEAR].values  # Pop data used in fitting


def get_prq_parameters(result_print=False, plot=False):
    years = MAJOR_HISTORICAL_FLOODS
    flood_ser = get_actual_water_series(SERVEY_START_YEAR, SERVEY_YEAR)
    damage_ser = pd.Series(index=years)
    for year in years:
        h = get_actual_levee_height(year)
        w = flood_ser[year]
        damage_ser.loc[year] = judge_damage(w, h)

    def ut(t, n, p, r):
        return n * np.e ** (-(p + r) * t)

    def vt(t, n, p, r, q):
        return (n * r / (p + r - q)) * (np.e ** ((-q) * t) - np.e ** (-(p + r) * t))

    def residuals(params, data):
        """
        :param params: consists of "p, r, q".
        :param data: The object ([U1, U2, ... U5, V1, V2, ... V5])
        years: Flooding year [1904? 1946? 1964? 1981? 2012?]
        """
        p, r, q,  = params["p"], params["r"], params["q"]
        u_list, v_list = [], []
        influenced_ser = damage_ser * estimated_population.loc[years]
        for y in years:
            n = influenced_ser[y]  # Estimation of flooding's influence in each survey year.
            t_delta = years.max() - y
            if t_delta > EXPECTANCY:  # Comparing with life expectancy of the survey year.
                t = np.arange(years.max() - EXPECTANCY)
            else:
                t = np.arange(t_delta)
            u = ut(t, n, p, r).sum()
            v = vt(t, n, p, r, q).sum()
            u_list.append(u), v_list.append(v)
        model = np.array(u_list + v_list)
        return data - model

    def fit(u_ratio, v_ratio):
        fit_params = Parameters()
        fit_params.add(name="p", value=0.75, min=0)
        fit_params.add(name="r", value=0.1, min=0)
        fit_params.add(name="q", value=0.05, min=0)

        data_y = np.array(list(u_ratio) + list(v_ratio))
        minner = Minimizer(residuals, fit_params, fcn_args=[data_y])
        result = minner.minimize()
        if result_print:
            report_fit(result)
        return result

    def get_params():
        df = stats_fre_questionnaris(questionnaires, t_list=years)
        u, v, m = df['communicative'].values, df['cultural'].values, df['collective'].values
        n = len(questionnaires)  # Number of total valid questionairs
        ratio = actual_population[SERVEY_YEAR] / (4 * n)
        optimised = fit(u*ratio, v*ratio)
        p, q, r = [optimised.params[key].value for key in ['p', 'q', 'r']]
        return [p, q, r]

    def plot_show():
        p, q, r = get_params()
        t = np.arange(100)
        n = 1
        m = (n / (p + r - q)) * ((p - q) * np.e ** (-(p + r) * t) + r * np.e ** (-q * t))
        u = ut(t, n, p, r)
        v = vt(t, n, p, r, q)
        plt.plot(t, m, '--', label='collective memory')
        plt.plot(t, u, 'm', label='communicative memory')
        plt.plot(t, v, 'b', label='cultural memory')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    para = get_params()
    if plot:
        plot_show()
    return para


def simu_k(model, result_print=False, how='exp'):
    ser = get_actual_water_series(FIT_START_YEAR, FIT_END_YEAR)
    d_initial = d_mean[FIT_START_YEAR]

    def residual(params):
        k = params['k_'+model]
        d_osm = d_initial
        d_iudm = d_initial
        u, v, m = 0, 0, 0  # initial
        y_model = []
        for i in ser.index:
            w = ser[i]
            h = get_actual_levee_height(i)
            f = judge_damage(w, h)
            m, d_osm = osm([f, m, d_osm], steps=1, k_osm=k, how=how)
            u, v, d_iudm = iudm([f, u, v, d_iudm], steps=1, k_iudm=k, how=how)
            if i in t_arr:
                if model == 'osm':
                    y_model.append(d_osm)
                elif model == 'iudm':
                    y_model.append(d_iudm)
        return np.array(y_model) - p_arr

    def optimise():
        params = Parameters()
        params.add(name='k_'+model, value=0.05, min=0)
        minner = Minimizer(residual, params)
        result = minner.minimize().params
        return result

    optimised = optimise()

    if result_print:
        print("The result of {} is:".format(model))
        optimised.pretty_print()

    return optimised['k_'+model].value


def get_k(result_print=False):
    def exp_model(t, t0, n0, k):
        return n0 * np.e ** (k * (t - t0))

    def residuals(params, t, data):
        k = params['k']
        n0 = data[0]
        model = exp_model(t, t[0], n0, k)
        return model - data

    def second_residuals(params, t, data):
        ka, kb = params['k1'], params['k2']
        t1, t2 = t[t <= LEVEE_YEAR_1], t[t > LEVEE_YEAR_1]
        n0 = data[0]
        t2_0 = t1[-1]
        n1 = d_mean[t2_0]
        model_1 = exp_model(t1, t1[0], n0, ka)
        model_2 = exp_model(t2, t2_0, n1, kb)
        return np.array(list(model_1) + list(model_2)) - data

    def residuals_minimum(residual_func, params):
        minner = Minimizer(residual_func, params, fcn_args=[t_arr, p_arr])
        result = minner.minimize().params
        return result

    params_1 = Parameters()  # The first model
    params_1.add(name="k", value=0.05, min=0)  # Initial value setting
    result_1 = residuals_minimum(residuals, params_1)
    k1 = result_1['k'].value

    if result_print:
        print("The first model's parameter:")
        result_1.pretty_print()

    params_2 = Parameters()
    params_2.add(name="k1", value=0.05, min=0)
    params_2.add(name="k2", value=0.05, min=0)
    result_2 = residuals_minimum(second_residuals, params_2)
    k21, k22 = result_2['k1'].value, result_2['k2'].value

    if result_print:
        print("The second model's parameters:")
        result_2.pretty_print()

    return k1, k21, k22


def estimate_population_from_1904(plot=False):
    t_1904 = 1904
    k_1904 = get_k()[0]

    def exp_model(t, t0, n0, k):
        return n0 * np.e ** (k * (t - t0))

    def residual(params):
        n0 = params['population_in_1904']
        model = exp_model(t_arr, t_1904, n0, k_1904)
        return p_arr - model

    fit_parameter = Parameters()
    fit_parameter.add(name='population_in_1904', value=actual_population[1940], min=0)
    minner = Minimizer(residual, params=fit_parameter)
    result = minner.minimize().params
    print("Estimating population in 1904:")
    result.pretty_print()
    n_1904 = result['population_in_1904'].value
    t_estimate = np.arange(t_1904, SERVEY_YEAR)
    pop_ = exp_model(t_estimate, t_1904, n_1904, k_1904) * P_MEAN

    if plot:
        plt.plot(t_estimate, pop_, label="Estimated population in 1904 is {:.0f}".format(n_1904 * P_MEAN))
        plt.legend()
        plt.show()

    return pd.Series(pop_, index=t_estimate)


def do_main_simu(how='exp'):
    """
    param k is population growth rate
    :return:
    """
    k1, k21, k22 = get_k(result_print=True)
    k_osm = simu_k('osm', result_print=True, how=how)
    k_iudm = simu_k('iudm', result_print=True, how=how)
    k_dic = {'k1': k1,
             'k21': k21,
             'k22': k22,
             'osm': k_osm,
             'iudm': k_iudm}
    with open('data/k_dic.json', 'w') as f:
        json.dump(k_dic, f)
    print('Successfully stored different parameter "k" as a json file.')


def do_population_simu():
    estimate_pop = estimate_population_from_1904(plot=True)
    estimate_pop.to_csv('data/Population_estimated_ser.csv', header=False)
    print('Successfully stored estimated population data as a csv file.')


def do_pqr_simu():
    p, q, r = get_prq_parameters(result_print=True, plot=True)
    pqr_dic = {
        'p': p,
        'q': q,
        'r': r
    }
    with open('data/pqr_dic.json', 'w') as f:
        json.dump(pqr_dic, f)
    print('Successfully stored "p, q, and r" parameters as a json file.')


def fit_exp_decay(bounder=True):
    # influenced of each major flood
    major_floods["relative_level"] = major_floods['flood_level'] - W_MIN

    # collective memory in ratio as a y_data
    ratio_collective = stats_fre_questionnaris(questionnaires, MAJOR_HISTORICAL_FLOODS, normalize=True)['collective']

    def exp_decay(t1, t0, n0, k):
        return n0 * np.exp(-k * (t1 - t0))

    def residual(params):
        n0, k = params['n'], params['k']
        influence = n0 * major_floods['relative_level']
        y_model = []
        for year in MAJOR_HISTORICAL_FLOODS:
            y_model.append(exp_decay(t1=SERVEY_YEAR, t0=year, n0=influence[year], k=k))
        return np.array(y_model) - ratio_collective

    fit_params = Parameters()
    if bounder:
        fit_params.add(name='n', value=1, min=0, max=1)
    else:
        fit_params.add(name='n', value=22, min=0)
    fit_params.add(name='k', value=0.015, min=0)
    minner = Minimizer(residual, fit_params)
    result = minner.minimize()
    report_fit(result)
    return result.params


if __name__ == '__main__':
    do_main_simu()
    # do_population_simu()
    # do_pqr_simu()
    # exp_decay_params = fit_exp_decay(plot=True, bounder=True)
    # get_prq_parameters(result_print=True, plot=True)
    pass
