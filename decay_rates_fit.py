#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/31 23:10
"""

from matplotlib import pyplot as plt
from lmfit import Parameters, Minimizer, report_fit
import seaborn as sns

from data_process import *
from collective_memory import judge_damage
from model_test import fit_nash, goodness_of_fit, adjusted_goodness
from repeat import repeat_simulating, one_of_simulating

fit_result = pd.DataFrame(columns=['data', 'vary', 'bound', 'x_col', 'y_col', 'params', 'nash', 'r2', 'adj_r2'])
VARY = True
BOUND = False


def get_data(kind):
    index = list(MAJOR_HISTORICAL_FLOODS) + [2018]
    flood = pd.DataFrame(index=index)

    datasets = {'all': questionnaires, 'farm': questionnaires[questionnaires['farm']],
                'off-farm': questionnaires[questionnaires['farm'] == False]}

    data = datasets[kind]

    for y in MAJOR_HISTORICAL_FLOODS:
        flood.loc[y, 'relative_level'] = major_floods['flood_level'].loc[y] - W_MIN
        flood.loc[y, 'exceeding_level'] = major_floods.loc[y, 'flood_level'] - get_actual_levee_height(y)
        flood.loc[y, 'damage'] = judge_damage(major_floods.loc[y, 'flood_level'], get_actual_levee_height(y))
        flood.loc[y, 'pop'] = estimated_population[y] * flood.loc[y, 'damage']
    flood.loc[2018, :] = np.zeros(len(flood.columns))

    memory = stats_fre_questionnaris(data, MAJOR_HISTORICAL_FLOODS, normalize=True)
    memory['sum'] = memory['communicative'] + memory['cultural']
    return memory, flood


def nomarlization(x):
    return (x - x.min()) / (x.max() - x.min())


def exp_decay_model(kind):
    memory, flood = get_data(kind)

    # Fitting
    def fit_exp_decay(flood_col, memory_col, bounder=False, vary=True):
        x_data = flood[flood_col]
        y_data = memory[memory_col]

        def exp_decay(t1, t0, n, mius):
            return n * (1 - mius) ** (t1 - t0)

        def residual(params, x, y):
            n, miuis = params['n'], params['k']
            influences = n * nomarlization(x)
            y_ = []
            for year in MAJOR_HISTORICAL_FLOODS:
                y_.append(exp_decay(t1=SERVEY_YEAR, t0=year, n=influences[year], mius=miuis))
            return np.array(y_) - y

        fit_params = Parameters()
        if bounder:
            fit_params.add(name='n', value=1, min=0, max=1)
        else:
            fit_params.add(name='n', value=0.5, min=0, vary=vary)
        fit_params.add(name='k', value=0.015, min=0)
        minner = Minimizer(residual, fit_params, fcn_args=[x_data, y_data])
        result = minner.minimize()
        report_fit(result)

        colors = ['b', 'g', 'r', 'c', 'm']
        k, n0 = [result.params[key].value for key in ['k', 'n']]
        influence = n0 * nomarlization(x_data).loc[MAJOR_HISTORICAL_FLOODS]
        y_model = []

        # plotting
        plt.scatter(x=influence.index, y=influence.values, color='w', edgecolors=colors)
        plt.scatter(x=[SERVEY_YEAR] * 5, y=y_data, color=colors)

        for i in range(len(MAJOR_HISTORICAL_FLOODS)):
            start_year = MAJOR_HISTORICAL_FLOODS[i]
            t_slices = np.arange(start_year, SERVEY_YEAR)
            y_func = exp_decay(t_slices, start_year, influence[start_year], k)
            plt.plot(t_slices, y_func, '--', c=colors[i])
            y_model.append(y_func[-1])
        plt.title(kind)
        plt.show()

        plt.scatter(y_data, y_model, label='forecast by model')
        plt.plot(y_data, y_data, label='1:1 line')
        plt.title(kind)
        plt.show()

        nash = fit_nash(y_data, y_model)
        r2 = goodness_of_fit(y_data, y_model)
        adj_r2 = adjusted_goodness(y_data, y_model)
        print("nash is {:.3f}, r^2 is {:.3f}, adjusted r2 is {:.3f}.".format(nash, r2, adj_r2))
        re = {
            'data': kind,
            'vary': vary,
            'bound': bounder,
            'x_col': flood_col,
            'y_col': memory_col,
            'params': "n = {:.3f}, k = {:.3f}".format(n0, k),
            'nash': "{:.3f}".format(nash),
            'r2': "{:.3f}".format(r2),
            'adj_r2': "{:.3f}".format(adj_r2)
        }
        return k, re

    miu_s, setting = fit_exp_decay('relative_level', 'collective', bounder=BOUND, vary=VARY)

    with open('data/{}_miu_s.json'.format(kind), 'w') as f:
        json.dump(miu_s, f)
    print('Successfully stored "miu_s" parameter as a json file.')


def iudm_decay_model(kind):
    memory, flood = get_data(kind)

    def fit_prq_decay(flood_col, memory_col, bounder=False, vary=True):
        x_data = flood[flood_col]
        u_data = list(memory['communicative'].values)
        v_data = list(memory['cultural'].values)
        m_data = list(memory[memory_col].values)
        y_data = np.array(u_data + v_data + m_data)

        # Fitting
        def ut(t, n, p, r):
            return n * np.exp(-(p + r) * t)

        def vt(t, n, p, r, q):
            return (n * r / (p + r - q)) * (np.exp((-q) * t) - np.e ** (-(p + r) * t))

        def mt(t, n, p, r, q):
            return (n / (p + r - q)) * ((p - q) * np.exp(-(p + r) * t) + r * np.exp(-q * t))

        def residuals(parameters, x, y):
            n, p, r, q = parameters['n'], parameters['p'], parameters['r'], parameters['q']
            influenced = n * nomarlization(x)
            communicative, cultural, collective = [], [], []
            for year in MAJOR_HISTORICAL_FLOODS:
                t = SERVEY_YEAR - year
                n = influenced[year]
                collective.append(mt(t, n, p, r, q))
                communicative.append(ut(t, n, p, r))
                cultural.append(vt(t, n, p, r, q))
            y_ = np.array(communicative + cultural + collective)
            return y - y_

        fit_params = Parameters()
        if bounder:
            fit_params.add(name='n', value=1, min=0, max=1, vary=vary)
        else:
            fit_params.add(name='n', value=1, min=0, vary=vary)
        fit_params.add(name="p", value=0.05, max=1)
        fit_params.add(name="r", value=0.03, min=0, max=1)
        fit_params.add(name="q", value=0.05, min=0, max=1)

        minner = Minimizer(residuals, fit_params, fcn_args=[x_data, y_data])
        result = minner.minimize()
        report_fit(result)

        colors = ['b', 'g', 'r', 'c', 'm']
        n0, param_p, param_q, param_r = [result.params[key].value for key in ['n', 'p', 'q', 'r']]
        influence = n0 * nomarlization(x_data).loc[MAJOR_HISTORICAL_FLOODS]
        u_list, v_list, m_list = [], [], []

        # plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
        ax1.scatter(x=influence.index, y=influence.values, color=colors, marker='o')
        ax1.scatter(x=[SERVEY_YEAR] * 5, y=memory[memory_col], color=colors)

        ax2.scatter(x=influence.index, y=influence.values, color=colors, marker='o')
        ax2.scatter(x=[SERVEY_YEAR] * 5, y=memory['communicative'], color=colors)

        ax3.scatter(x=influence.index, y=[0] * 5, color=colors, marker='o')
        ax3.scatter(x=[SERVEY_YEAR] * 5, y=memory['cultural'], color=colors)

        for i in range(len(MAJOR_HISTORICAL_FLOODS)):
            start_year = MAJOR_HISTORICAL_FLOODS[i]
            t_slices = np.arange(start_year, SERVEY_YEAR)
            u_model = ut(t_slices - start_year, influence[start_year], param_p, param_r)
            v_model = vt(t_slices - start_year, influence[start_year], param_p, param_r, param_q)
            m_model = mt(t_slices - start_year, influence[start_year], param_p, param_r, param_q)
            u_list.append(u_model[-1])
            v_list.append(v_model[-1])
            m_list.append(m_model[-1])
            ax1.plot(t_slices, m_model, '--', c=colors[i])
            ax2.plot(t_slices, u_model, '-.', c=colors[i])
            ax3.plot(t_slices, v_model, '-.', c=colors[i])
        for ax in [ax1, ax2, ax3]:
            ax.set_ylim(-0.05, 1.5)
            ax.set_title(kind)
        plt.show()

        plt.scatter(u_data, u_list, label='Communicative forecast by model', c='b')
        plt.scatter(v_data, v_list, label='Cultural forecast by model', c='g')
        plt.scatter(m_data, m_list, label='Collective forecast by model', c='r')
        plt.plot(y_data, y_data, label='1:1 line')
        plt.legend()
        plt.title(kind)
        plt.show()

        y_model = np.array(u_list + v_list + m_list)
        r2 = goodness_of_fit(y_data, y_model)
        nash = fit_nash(y_data, y_model)
        adj_r2 = adjusted_goodness(y_data, y_model)
        print("nash is {:.3f}, r^2 is {:.3f}, adjusted r2 is {:.3f}.".format(nash, r2, adj_r2))

        re = {
            'data': kind,
            'vary': vary,
            'bound': bounder,
            'x_col': flood_col,
            'y_col': memory_col,
            'params': "n = {:.3f}, p = {:.3f}, r = {:.3f}, q = {:.3f}".format(n0, param_p, param_r, param_q),
            'nash': "{:.3f}".format(nash),
            'r2': "{:.3f}".format(r2),
            'adj_r2': "{:.3f}".format(adj_r2)
        }

        return (param_p, param_r, param_q), re

    params, setting = fit_prq_decay('relative_level', 'sum', bounder=BOUND, vary=VARY)

    pqr_dic = {
        'p': params[0],
        'q': params[2],
        'r': params[1]
    }
    with open('data/{}_pqr_dic.json'.format(kind), 'w') as f:
        json.dump(pqr_dic, f)
    print('Successfully stored "p, q, and r" parameters as a json file.')


def do_diff_pqr_and_mius_simulations():
    for kind in ['all', 'farm', 'off-farm']:
        exp_decay_model(kind)
        iudm_decay_model(kind)


if __name__ == '__main__':
    do_diff_pqr_and_mius_simulations()
    pass
