#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/4/3 11:34
"""
from decay_rates_fit import *
from simulating import one_of_simulating


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    sns.boxplot(x='used dataset', y=osm_col, data=df, ax=ax1)
    sns.boxplot(x='used dataset', y=iudm_col, data=df, ax=ax2)
    ax1.set_xlabel('osm model')
    ax2.set_xlabel('iudm model')
    plt.show()


def r_sensibility(initial_kind='all', interval=0.2):
    p, r, q = [get_last_pqr(initial_kind)[k] for k in ['p', 'r', 'q']]
    result_list = []
    for r in np.arange(0, 1, interval):
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


if __name__ == '__main__':
    all_results = get_all_repeated_simu_results(years=100, times=100, how='linear', k=0.03)
    boxes_of_data_kinds(all_results, 'm', 'u+v')
    plt.title('linear')
    plt.show()

    boxes_of_data_kinds(all_results, 'loss_osm', 'loss_iudm')
    plt.show()
    pass
