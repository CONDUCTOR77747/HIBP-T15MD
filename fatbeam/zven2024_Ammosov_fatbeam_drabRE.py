# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:41:50 2023

@author: conductor

Picture with whole fatbeam and zoomed slits and ionization zone

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
# from hibpplotlib import set_axes_param
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

def set_axes_param(ax, xlabel, ylabel, isequal=True, fontsize=14):
    '''
    format axes
    '''
    # ax.grid(True)
    # ax.grid(which='major', color='tab:gray')  # draw primary grid
    ax.minorticks_on()  # make secondary ticks on axes
    # draw secondary grid
    # ax.grid(which='minor', color='tab:gray', linestyle=':')
    ax.xaxis.set_tick_params(width=2, labelsize=fontsize)  # increase tick size
    ax.yaxis.set_tick_params(width=2, labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if isequal:
        ax.axis('equal')

# mpl.rcParams['figure.dpi'] = 100

# Set font properties
font = {'fontname': 'Times New Roman'}
font2 = {'labelfontfamily': 'Times New Roman'}
# Set font size
fontsize = 20

plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 150

n_slit = 3
geom = geomT15
tr = traj_fat_beam

alpha = 0.5

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax2 = inset_axes(ax1, width="30%", height=1.5, loc=1)

set_axes_param(ax1, 'x, m', 'y, m', isequal=False)
set_axes_param(ax2, 'z, m', 'y, m', isequal=False)

# ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
#               .format(tr[0].Ebeam, tr[0].U['A2'], Btor, Ipl), **font)

# get number of slits
n_slits = geom.plates_dict['an'].n_slits
# set color cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors[:n_slits]

if n_slit == 'all':
    slits = range(n_slits)
else:
    slits = [n_slit]

# plot trajectories
for tr in traj_fat_beam:
    # plot primary trajectory
    tr.plot_prim(ax1, axes='XY', color='k', full_primary=False)
    tr.plot_prim(ax2, axes='ZY', color='k', full_primary=False)
    
    # plot first point
    ax1.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')
    ax2.plot(tr.RV0[0, 2], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white', markersize=7)

    # plot secondaries
    for i in slits:
        c = colors[i]
        if tr.RV_sec_toslits[i] is None:
            continue
        for fan_tr in tr.RV_sec_toslits[i]:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
            # ax2.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)
            
            # plot zones
            ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            # ax2.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
            #          markerfacecolor='white')
            
# plot geometry: T-15MD wall, plates, analyzer and etc.
geom.plot(ax1, axes='XY', plot_aim=False, plot_analyzer=True)
geom.plot(ax2, axes='ZY', plot_aim=False, plot_analyzer=True)

# Set font properties for ax1
labelpad = -8.5
ax1.set_xlabel('x, m', **font, fontsize=fontsize, labelpad=labelpad)
ax1.set_ylabel('y, m', **font, fontsize=fontsize, labelpad=labelpad)

# Set font properties for tick labels
ax1.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize)  # increase tick size
ax1.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize)

fig1.subplots_adjust(left=0.35, bottom=0.2, right=0.75, top=0.9)

ax1.set_xlim(1.51, 4.8)
ax1.set_ylim(-0.25, 1.8)


# Set font properties for ax2
labelpad = -8.5
ax2.set_xlabel('x, m', **font, fontsize=fontsize, labelpad=labelpad)
ax2.set_ylabel('y, m', **font, fontsize=fontsize, labelpad=labelpad)

# Set font properties for tick labels
ax2.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize)  # increase tick size
ax2.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize)

fig2.subplots_adjust(left=0.4, bottom=0.2, right=0.7, top=0.9)

ax2.set_xlim(0.11, 0.145)
ax2.set_ylim(1.675, 1.705)

ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

plt.show()

plt.close(fig2)
