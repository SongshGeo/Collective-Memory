#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/4/3 12:43
"""
import pandas as pd
from collective_memory import main_function
from generate_floods import generate_random_flood_series
from scipy.stats import ttest_rel
from repeated_simulating import repeat_simulating


def one_of_simulating(years, k, kind, fre=0., how='exp', random_state=None, technology=False):
    ser = generate_random_flood_series(years=years, fre=fre, random_state=random_state)
    df = main_function(ser, how=None, k_iudm=k, k_osm=k, pop_growth=how, kind=kind)
    if technology:
        df2 = main_function(ser, how='enhance', k_osm=k, k_iudm=k, pop_growth=how)
    losses = df.loc[:, ['loss_osm', 'loss_iudm']]
    losses.dropna(axis='rows', inplace=True)
    print(ttest_rel(losses['loss_osm'], losses['loss_iudm']))
    return df


def get_all_repeated_simu_results(years, k, times, how, fre=0.):
    data_list = []
    for kind in ['all', 'farm', 'off-farm']:
        data = repeat_simulating(kind=kind, years=years, times=times, k=k, fre=fre, how=how)
        data['used dataset'] = kind
        data_list.append(data)
    return pd.concat(data_list)


if __name__ == '__main__':
    simu_result = one_of_simulating(years=100, k=0.03, kind='all', how='exp', random_state=1, technology=False)
    osm = simu_result['loss_osm'].mean()
    iudm = simu_result['loss_iudm'].mean()
    percentage = (osm - iudm) / osm
    pass
