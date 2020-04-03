#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/4/3 11:34
"""
from decay_rates_fit import *
from repeated_simulating import repeat_simulating
from simulating import one_of_simulating
from scipy.stats import ttest_rel


def get_all_repeated_simu_results(years, k, times, how, fre=0.):
    data_list = []
    for kind in ['all', 'farm', 'off-farm']:
        data = repeat_simulating(kind=kind, years=years, times=times, k=k, fre=fre, how=how)
        data['used dataset'] = kind
        data_list.append(data)
    return pd.concat(data_list)


def get_all_simu_results(years, k, fre, how, random_state=1):
    data_list = []
    for kind in ['all', 'farm', 'off-farm']:
        df = one_of_simulating(years=years, k=k, fre=fre, how=how, kind=kind, random_state=random_state)
        df['kind'] = kind
        data_list.append(df)
    return pd.concat(data_list)


def boxes_of_data_kinds(df, osm_col, iudm_col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey='all')
    sns.boxplot(x='used dataset', y=osm_col, data=df, ax=ax1)
    sns.boxplot(x='used dataset', y=iudm_col, data=df, ax=ax2)
    ax1.set_xlabel('osm model')
    ax2.set_xlabel('iudm model')
    plt.show()


def r_sensibility(initial_kind='all', interval=0.2):
    p, r, q = [get_last_pqr(initial_kind)[k] for k in ['p', 'r', 'q']]
    result_list = []
    for r in np.arange(0.1, 1, interval):
        pqr_dic = {
            'p': p,
            'q': r,
            'r': q
        }
        with open('data/{}_pqr_dic.json'.format(initial_kind), 'w') as f:
            json.dump(pqr_dic, f)
        print('Successfully stored "p, q, and r" parameters as a json file.')

        result = repeat_simulating(kind=initial_kind, years=100, k=0.03)
        result['used r'] = str(r)
        result_list.append(result)
    data = pd.concat(result_list)
    sns.boxplot(x='used r', y='loss_iudm', data=data)
    plt.show()
    iudm_decay_model(initial_kind)
    plt.show()


def compare_models_ttest(df):
    kinds = ['all', 'farm', 'off-farm']

    grouped = df.groupby('used dataset')
    osm, iudm = [], []
    for kind, data in grouped:
        osm.append((kind, data['loss_osm']))
        iudm.append((kind, data['loss_iudm']))
    osm_list = [list([osm[a], osm[b]]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    iudm_list = [list([iudm[a], iudm[b]]) for a, b in [(0, 1), (0, 2), (1, 2)]]

    # osm
    osm_df = pd.DataFrame(index=kinds, columns=kinds)
    for a, b in osm_list:
        kind_1, data_1 = a
        kind_2, data_2 = b
        osm_df.loc[kind_1, kind_2] = ttest_rel(data_1, data_2)

    iudm_df = pd.DataFrame(index=kinds, columns=kinds)
    for a, b in iudm_list:
        kind_1, data_1 = a
        kind_2, data_2 = b
        iudm_df.loc[kind_1, kind_2] = ttest_rel(data_1, data_2)

    return osm_df, iudm_df


if __name__ == '__main__':
    all_results = get_all_repeated_simu_results(years=100, times=100, how='exp', k=0.03)
    boxes_of_data_kinds(all_results, 'm', 'u+v')
    plt.show()

    boxes_of_data_kinds(all_results, 'loss_osm', 'loss_iudm')
    plt.show()

    r_sensibility()
    pass
