#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/31 9:25
"""
from collective_memory import main_function
from generate_floods import generate_random_flood_series

from matplotlib import pyplot as plt
from scipy.stats import ttest_rel
import numpy as np
import pandas as pd


def repeat_simulating(years, k, kind, times=100, fre=0., how='exp'):
    flag = 0
    df = pd.DataFrame(index=range(times))
    while flag < times:
        ser = generate_random_flood_series(years=years, fre=fre, random_state=flag)
        simu = main_function(ser, kind=kind, how=None, k_iudm=k, k_osm=k, pop_growth=how, print_params=False)
        for col in ['m', 'u', 'v']:
            df.loc[flag, col] = simu[col].mean()
        for col in ['loss_iudm', 'loss_osm']:
            df.loc[flag, col] = simu[col].sum()
        flag += 1
    df['u+v'] = df['u'] + df['v']
    return df


def boxplot(osm_list, iudm_list, labels, ax):
    widths = 0.4
    position = np.array([1, 3, 5])
    osm = ax.boxplot(osm_list, positions=position, whis=1.6, widths=widths, sym="o", patch_artist=True)
    iudm = ax.boxplot(iudm_list, positions=position+widths, whis=1.6, widths=widths, sym='o', patch_artist=True)
    for patch in osm["boxes"]:
        patch.set_facecolor(color="b")
    for patch in iudm["boxes"]:
        patch.set_facecolor(color="g")
    ax.set_xticks(position + widths / 2, labels)
    ax.set_xlim(0.4, 6)
    ax.set_xticklabels(labels)
    plt.grid(axis="y", ls=":", lw=1, alpha=0.4)


def print_average_compare(df, kind, label):
    osm = df['loss_osm'].mean()
    iudm = df['loss_iudm'].mean()
    percentage = (osm - iudm) / osm
    print("Simulation in {} dataset, {}".format(kind, label))
    print("Total losses by osm is {:.3f}, by iudm is {:.3f}, osm exceed iudm {:.3%}.".format(osm, iudm, percentage))


def repeating_different_frequency(how, kind, years, k, ax, times=100):
    labels = ["+0%", "+20%", "+40%"]
    fres = [0., 0.2, 0.4]
    osm_list, iudm_list = [], []
    for i in range(len(fres)):
        fre = fres[i]
        df = repeat_simulating(how=how, kind=kind, years=years, k=k, times=times, fre=fre)
        osm_list.append(df['loss_osm'])
        iudm_list.append(df['loss_iudm'])
        print_average_compare(df=df, kind=kind, label=labels[i])
    boxplot(osm_list, iudm_list, labels, ax)


def repeat_different_duration(how, kind, k, ax, times=100):
    years = [50, 100, 200]
    osm_list, iudm_list = [], []
    labels = ["50 years", "100 years", "200 years"]
    for i in range(len(years)):
        length = years[i]
        df = repeat_simulating(how=how, kind=kind, years=length, k=k, times=times)
        osm_list.append(df['loss_osm'])
        iudm_list.append(df['loss_iudm'])
        print_average_compare(df=df, kind=kind, label=labels[i])
    boxplot(osm_list, iudm_list, labels, ax)


def plot_boxes_to_show_osm_iudm_differences(how, kind, k):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    repeating_different_frequency(how=how, kind=kind, years=100, k=k, ax=ax1)
    repeat_different_duration(how=how, kind=kind, k=k, ax=ax2)
    plt.title("show differences by {}".format(kind))
    plt.show()


if __name__ == '__main__':
    plot_boxes_to_show_osm_iudm_differences(how='exp', kind='all', k=.03)
    pass
