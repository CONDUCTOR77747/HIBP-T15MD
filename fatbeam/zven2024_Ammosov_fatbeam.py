# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:41:50 2023

@author: conductor

Picture with whole fatbeam and zoomed slits and ionization zone

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from hibpplotlib import set_axes_param
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

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

# fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
fig1, ax1 = plt.subplots()
# fig2, axins = plt.subplots()
# fig3, axins_slits = plt.subplots()

# Make the zoom-in plot:
axins = zoomed_inset_axes(ax1, 30, loc=1) # zoom = 30
axins_slits = zoomed_inset_axes(ax1, 8, loc=4) # zoom = 2


set_axes_param(ax1, 'x, m', 'y, m', isequal=False)
set_axes_param(axins, 'x, m', 'y, m', isequal=False)
set_axes_param(axins_slits, 'x, m', 'y, m', isequal=False)

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
    tr.plot_prim(axins, axes='XY', color='k', full_primary=False, alpha=alpha)
    tr.plot_prim(axins_slits, axes='XY', color='k', full_primary=False)
    
    # plot first point
    ax1.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')
    axins.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')
    axins_slits.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')

    # plot secondaries
    for i in slits:
        c = colors[i]
        if tr.RV_sec_toslits[i] is None:
            continue
        for fan_tr in tr.RV_sec_toslits[i]:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
            axins.plot(fan_tr[:, 0], fan_tr[:, 1], color=c, alpha=alpha)
            axins_slits.plot(fan_tr[:, 0], fan_tr[:, 1], color=c, alpha=alpha)
            
            # plot zones
            ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            axins.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            axins_slits.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            
# plot geometry: T-15MD wall, plates, analyzer and etc.
geom.plot(ax1, axes='XY', plot_aim=False, plot_analyzer=True)
geom.plot(axins_slits, axes='XY', plot_aim=False, plot_analyzer=True)

# Set font properties for axes labels
labelpad = -8.5
ax1.set_xlabel('x, m', **font, fontsize=fontsize, labelpad=labelpad)
ax1.set_ylabel('y, m', **font, fontsize=fontsize, labelpad=labelpad)

# Set font properties for tick labels
ax1.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize)  # increase tick size
ax1.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize)

fig1.subplots_adjust(left=0.35, bottom=0.2, right=0.7, top=0.9)

ax1.set_xlim(1.51, 4.8)
ax1.set_ylim(-0.63, 2.14)



# axins ion zones
axins.set_xlim(1.785, 1.82)
axins.set_ylim(0.77, 0.80)
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.0")

fontsize_axins = 25

axins.set_xlabel('x, m', **font, fontsize=fontsize_axins)
axins.set_ylabel('y, m', **font, fontsize=fontsize_axins)
# Set font properties for tick labels
axins.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_axins)  # increase tick size
axins.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_axins)

axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)


# axins_slits
axins_slits.set_xlim(4.06, 4.17)
axins_slits.set_ylim(0.73, 0.86)
mark_inset(ax1, axins_slits, loc1=2, loc2=4, fc="none", ec="0.0")

axins_slits.set_xlabel('x, m', **font, fontsize=fontsize_axins)
axins_slits.set_ylabel('y, m', **font, fontsize=fontsize_axins)
# Set font properties for tick labels
axins_slits.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_axins)  # increase tick size
axins_slits.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_axins)

axins_slits.get_xaxis().set_visible(False)
axins_slits.get_yaxis().set_visible(False)

plt.show()
