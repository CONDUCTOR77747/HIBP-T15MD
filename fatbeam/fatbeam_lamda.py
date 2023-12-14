# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:06:09 2023

@author: ammosov_yam
"""

import os
import wire
import pylab
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import hibplib as hb
import hibpcalc.fields as fields
from oscillation import CoordConverterDShapeSimple


def set_axes_param(ax, xlabel, ylabel, isequal=True, fontsize=14):
    '''
    format axes
    '''
    ax.grid(True)
    ax.grid(which='major', color='tab:gray')  # draw primary grid
    ax.minorticks_on()  # make secondary ticks on axes
    # draw secondary grid
    ax.grid(which='minor', color='tab:gray', linestyle=':')
    ax.xaxis.set_tick_params(width=2, labelsize=fontsize)  # increase tick size
    ax.yaxis.set_tick_params(width=2, labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if isequal:
        ax.axis('equal')


def plot_lam(traj_list, Ebeam='all', slits=range(7)):
    '''
    plot SV size along trajectory (lambda) vs UA2 and vs rho
    '''
    # plotting params
    fig1, ax1 = plt.subplots()
    # fig1, ax2 = plt.subplots()
    set_axes_param(ax1, 'UA2 (kV)', r'$\lambda$ (mm)')
    # set_axes_param(ax2, r'$\rho', r'$\lambda$ (mm)')

    if Ebeam == 'all':
        equal_E_list = np.array([tr.Ebeam for tr in traj_list])
        equal_E_list = np.unique(equal_E_list)
    else:
        equal_E_list = np.array([float(Ebeam)])

    for Eb in equal_E_list:
        for i_slit in slits:
            UA2_list = []
            rho_list = []
            lam_list = []
            for tr in traj_list:
                if tr.Ebeam == Eb and tr.ion_zones[i_slit].shape[0] > 0:
                    UA2_list.append(tr.U['A2'])
                    # rho_list.append(rho_interp(tr.RV_sec[0, :3])[0])
                    lam_list.append(np.linalg.norm(tr.ion_zones[i_slit][0]
                                                   - tr.ion_zones[i_slit][-1])*1000)
            ax1.plot(UA2_list, lam_list, '-o', label='slit ' + str(i_slit+1))
            # ax2.plot(rho_list, lam_list, '-o', label='slit ' + str(i_slit+1))
    ax1.legend()
    # ax2.legend()
    plt.show()
    
    return lam_list
    
lam_list = plot_lam(traj_fat_beam, Ebeam='260', slits=[3])