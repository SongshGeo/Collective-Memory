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


def repeating_different_frequency(kind, years, k, ax, times=100):
    labels = ["+0%", "+20%", "+40%"]
    fres = [0., 0.2, 0.4]
    osm_list, iudm_list = [], []
    for i in range(len(fres)):
        fre = fres[i]
        df = repeat_simulating(kind=kind, years=years, k=k, times=times, fre=fre)
        osm_list.append(df['loss_osm']/years)
        iudm_list.append(df['loss_iudm']/years)
    boxplot(osm_list, iudm_list, labels, ax)


def repeat_different_duration(kind, k, ax, times=100):
    years = [50, 100, 200]
    osm_list, iudm_list = [], []
    labels = ["50 years", "100 years", "200 years"]
    for i in range(len(years)):
        length = years[i]
        df = repeat_simulating(kind=kind, years=length, k=k, times=times)
        osm_list.append(df['loss_osm']/length)
        iudm_list.append(df['loss_iudm']/length)
    boxplot(osm_list, iudm_list, labels, ax)


def plot_boxes_to_show_osm_iudm_differences(kind, title="show differences between models"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    repeating_different_frequency(kind=kind, years=100, k=0.03, ax=ax1)
    repeat_different_duration(kind='all', k=0.03, ax=ax2)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    pass
