#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  :   Shuang Song
Beijing Normal University
@Contact :   SongshGeo@Gmail.com
@Time    :   2020/3/20 9:58
"""


# Initialization of plot setting
def plot_initial_sets():
    from matplotlib import rc
    import matplotlib as mpl
    rc('text', usetex=True)
    rc('text.latex', preamble=r"""
    \usepackage[eulergreek]{sansmath}\sansmath
    \renewcommand{\rmdefault}{phv} % Arial
    \renewcommand{\sfdefault}{phv} % Arial
    """)
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["axes.unicode_minus"] = False


if __name__ == '__main__':
    pass
