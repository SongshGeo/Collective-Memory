#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/4/3 12:43
"""
from collective_memory import main_function
from generate_floods import generate_random_flood_series
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel


def plot(df):
    t_arr = df['y'].values
    w, h, loss_osm, loss_iudm, u, v, m = [df[col].values for col in ['w', 'h', 'loss_osm', 'loss_iudm', 'u', 'v', 'm']]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(311)
    ax1.bar(t_arr, w, color="b", label="High water level (W)")
    ax1.plot(t_arr, h, "--", color="k", label="Height of levee (H)")
    ax1.set_xlim(t_arr.min(), t_arr.max())

    plt.ylabel("( m )")
    plt.legend(loc=1)
    plt.subplot(323)
    plt.plot(t_arr, u+v, "-", label="M")
    plt.plot(t_arr, u, "--", color="c", label="U")
    plt.plot(t_arr, v, "--", color="m", label="V")
    plt.ylim(0, 0.3)
    plt.xlim(t_arr.min(), t_arr.max())
    plt.legend(loc=1)

    plt.subplot(325)
    markerline, stemlines, baseline = plt.stem(t_arr, loss_osm, linefmt="-.", markerfmt="o", label="losses(F × D)",
                                               use_line_collection=True)
    plt.setp(baseline, color="r", linewidth=2)
    plt.setp(markerline, color="r")
    plt.setp(stemlines, color="m")
    plt.legend(loc=1)
    plt.ylim(0, 0.04)
    plt.xlim(t_arr.min(), t_arr.max())
    plt.xlabel("Simulated by model integrated the UDM")

    plt.subplot(324)
    plt.plot(t_arr, m, "-", label="M")
    plt.ylim(0, 0.3)
    plt.xlim(t_arr.min(), t_arr.max())
    plt.legend(loc=1)

    plt.subplot(326)
    markerline, stemlines, baseline = plt.stem(t_arr, loss_iudm, linefmt="-.", markerfmt="o",
                                               use_line_collection=True, label="losses(F × D)")
    plt.setp(baseline, color="r", linewidth=2)
    plt.setp(markerline, color="r")
    plt.setp(stemlines, color="m")
    plt.xlim(t_arr.min(), t_arr.max())
    plt.xlabel("Simulation by traditional model")
    plt.ylim(0, 0.04)
    plt.legend(loc=1)
    plt.show()


def one_of_simulating(years, k, kind, fre=0., how='exp', random_state=None, technology=False):
    ser = generate_random_flood_series(years=years, fre=fre, random_state=random_state)
    df = main_function(ser, how=None, k_iudm=k, k_osm=k, pop_growth=how, kind=kind)
    plot(df)
    if technology:
        df2 = main_function(ser, how='enhance', k_osm=k, k_iudm=k, pop_growth=how)
        plot(df2)
    losses = df.loc[:, ['loss_osm', 'loss_iudm']]
    losses.dropna(axis='rows', inplace=True)
    print(ttest_rel(losses['loss_osm'], losses['loss_iudm']))
    return df


if __name__ == '__main__':
    simu_result = one_of_simulating(years=100, k=0.03, kind='all', how='linear', random_state=1, technology=False)
    pass
