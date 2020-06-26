#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/20 12:46
"""

import pandas as pd
import numpy as np
import json
import os

# Water level data
water_level = pd.read_csv('data/Water_data.csv', index_col=['year'])  # Water level of actual floodings.
W_MIN, W_MAX = 1135.3, 1141.5  # The minimum and maxmum water level
MAJOR_HISTORICAL_FLOODS = np.array([1904, 1946, 1964, 1981, 2012])
major_floods = water_level.loc[MAJOR_HISTORICAL_FLOODS]

# actual_population data
actual_population = pd.read_csv("data/Population_ser.csv", index_col=[0], header=None, encoding="GBK", squeeze=True)
pop = pd.read_csv('data/Population_data.csv', encoding='gbk')
estimated_population = pd.read_csv("data/Population_estimated_ser.csv", index_col=[0], header=None, squeeze=True)
P_MIN: float = 160 * 10000 * 2.5  # Pessitive estimation of the actual_population limits
P_MAX: float = 200 * 10000 * 2.5  # positive estimation of the actual_population limits
P_MEAN = (P_MIN + P_MAX) / 2

# Questionair data
SERVEY_START_YEAR = 1904
SERVEY_YEAR = 2018  # We do this servey in 2018
questionnaires = pd.read_csv("data/valid_data_in_English.csv", index_col=[0], encoding='utf8')
EXPECTANCY = 73.38  # Expectation of life in the servey year, study area.
datasets = {'all': questionnaires, 'farm': questionnaires[questionnaires['farm']],
            'off-farm': questionnaires[questionnaires['farm'] == False]}


def class_answers_type(answer, h, r):
    heard_of_answers = ["heard of somewhere", "heard from intimates"]  # These answers refer communicative momery
    records_answer = ['know from written records']  # This answer refers cultural memory
    answers = answer.split("; ")
    for answer in answers:
        answer = answer.strip()
        if answer in heard_of_answers:
            h.append(True)
            break
        elif answer in records_answer:
            r.append(True)
            break
    return h, r


def stats_fre_questionnaris(data, t_list, normalize=False):
    df = pd.DataFrame(index=t_list)
    collective_answers = ["heard of somewhere", "heard from intimates", 'know from written records', 'experienced']
    for year in t_list:
        heards, records = [], []
        data[str(year)].apply(class_answers_type, args=[heards, records])
        df.loc[year, 'communicative'] = np.array(heards).sum()
        df.loc[year, 'cultural'] = np.array(records).sum()
        df.loc[year, 'collective'] = data[str(year)].apply(lambda x: True if x in collective_answers else False).sum()
    if normalize:
        for col in df:
            df[col] = df[col] / len(data)
    df['sum'] = df['communicative'] + df['cultural']
    return df


def get_actual_water_series(start_year, end_year):
    ser = pd.Series(index=np.arange(start_year, end_year+1))
    for i in water_level.index:
        if i in ser.index:
            ser[i] = water_level.loc[i, 'flood_level']
        else:
            pass
    return ser


def get_actual_levee_height(y):
    h = 1135.3 if y <= 1981 else 1136.6 if y <= 2009 else 1137.34
    return h


def get_population():
    d_max = actual_population / P_MIN
    d_min = actual_population / P_MAX
    d_mean = actual_population * 2 / (P_MAX + P_MIN)
    return d_max, d_mean, d_min


def get_last_k():
    if os.path.exists('data/k_dic.json'):
        with open('data/k_dic.json', 'r') as f:
            k_dic = json.load(f)
    else:
        k_dic = {'k1': 0.032,
                 'k21': 0.033,
                 'k22': 0.029,
                 'osm': 0.0485,
                 'iudm': 0.0453}
    return k_dic


def get_last_pqr(data):
    file_name = 'data/{}_pqr_dic.json'.format(data)
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            pqr_dic = json.load(f)
    else:
        raise FileExistsError("{} not exists.".format(file_name))
    return pqr_dic


def get_last_mius(data):
    file_name = 'data/{}_miu_s.json'.format(data)
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            miu_s = json.load(f)
    else:
        raise FileExistsError("{} not exists.".format(file_name))
    return miu_s


def get_decay_params(data, model):
    file_name = 'data/{}_{}_params.json'.format(data, model)
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            params = json.load(f)
    else:
        raise FileExistsError("{} not exists.".format(file_name))
    return params


if __name__ == '__main__':
    pass
