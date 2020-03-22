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

flood = pd.read_csv('data/Water_data.csv', index_col=['year'])  # Water level of actual floodings.
W_MIN, W_MAX = 1135.3, 1141.5  # The minimum and maxmum water level

population = pd.read_csv("data/Population_ser.csv", index_col=[0], header=None, encoding="GBK", squeeze=True)
estimated_population = pd.read_csv("data/Population_estimated_ser.csv", index_col=[0], header=None, squeeze=True)
P_MIN: float = 160 * 10000 * 2.5  # Pessitive estimation of the population limits
P_MAX: float = 200 * 10000 * 2.5  # positive estimation of the population limits
P_MEAN = (P_MIN + P_MAX) / 2

# Questionair data
SERVEY_START_YEAR = 1904
SERVEY_YEAR = 2018  # We do this servey in 2018
queries = pd.read_csv('data/CM_data.csv', index_col=[0], encoding='gbk')
SERVEY_YEARS = np.array([1904, 1946, 1964, 1981, 2012])
EXPECTANCY = 73.38  # Expectation of life in the servey year, study area.


def class_answers_type(answer, h, r):
    heard_of_answers = ["有所耳闻", "周遭经历"]  # These answers refer communicative momery
    records_answer = ['书面记载']  # This answer refers cultural memory
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


def clean_questionaris(t_list):
    ser_u, ser_v = pd.Series(index=t_list), pd.Series(index=t_list)
    for year in t_list:
        heards, records = [], []
        queries[str(year)].apply(class_answers_type, args=[heards, records])
        ser_u.loc[year] = np.array(heards).sum()
        ser_v.loc[year] = np.array(records).sum()
    return ser_u.values, ser_v.values


questionair = {
    't': SERVEY_YEARS,  # Querying years.
    'u': clean_questionaris(SERVEY_YEARS)[0],  # Heard from others.
    'v': clean_questionaris(SERVEY_YEARS)[1],  # Knew from physical records.
    'n': len(queries)  # Total number of questionairs.
}


def get_actual_water_series(start_year, end_year):
    ser = pd.Series(index=np.arange(start_year, end_year+1))
    for i in flood.index:
        if i in ser.index:
            ser[i] = flood.loc[i, 'flood_level']
        else:
            pass
    return ser


def get_actual_levee_height(y):
    h = 1135.3 if y <= 1981 else 1136.6 if y <= 2009 else 1137.34
    return h


def get_population():
    d_max = population / P_MIN
    d_min = population / P_MAX
    d_mean = population * 2 / (P_MAX + P_MIN)
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


def get_last_pqr():
    if os.path.exists('pqr_dic.json'):
        with open('pqr_dic.json', 'r') as f:
            pqr_dic = json.load(f)
    else:
        pqr_dic = {
            'p': 0.0920,
            'r': 0.0142,
            'q': 0
        }
    return pqr_dic


if __name__ == '__main__':
    pass
