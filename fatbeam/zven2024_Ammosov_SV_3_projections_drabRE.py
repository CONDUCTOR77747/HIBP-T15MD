# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:41:50 2023

@author: conductor

Picture with SV in 3 projections

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from hibpplotlib import set_axes_param
import hibpplotlib as hbplotlib
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from oscillation import CoordConverterDShapeSimple
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
import copy

# mpl.rcParams['figure.dpi'] = 100

# Set font properties
font = {'fontname': 'Times New Roman'}
font2 = {'labelfontfamily': 'Times New Roman'}


plt.rcParams['savefig.dpi'] = 120
plt.rcParams['figure.dpi'] = 150

n_slit = 3
geom = geomT15

# get number of slits
n_slits = geom.plates_dict['an'].n_slits
# set color cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors[:n_slits]

# 3 proj fig

fig_sv3 = plt.figure()
ax1 = fig_sv3.add_subplot(2, 2, 1)
ax2 = fig_sv3.add_subplot(2, 2, 2)
ax3 = fig_sv3.add_subplot(2, 2, 3)
ax4 = fig_sv3.add_subplot(2, 2, 4, projection='3d')

axis = [(ax1, ax2), (ax3, ax4)]

# axis[1][1].axis('off')

# get coords
for tr in traj_fat_beam:

    # plot trajectories
    
    if n_slit == 'all':
        slits = reversed(range(n_slits))
    else:
        slits = [n_slit]
        
    # coords of ionization zones for each slit
    coords_list = []
    
    # plot sample volumes
    for i in slits:
        coords = np.empty([0, 3])
        coords_first = np.empty([0, 3])
        coords_last = np.empty([0, 3])
        for tr in traj_fat_beam:
            # skip empty arrays
            if tr.ion_zones[i] is None:
                continue
            if tr.ion_zones[i].shape[0] == 0:
                continue
            coords_first = np.vstack([coords_first, tr.ion_zones[i][0, 0:3]])
            coords_last = np.vstack([coords_last, tr.ion_zones[i][-1, 0:3]])
    
        if coords_first.shape[0] == 0:
            coords_list.append(None)
            print(f"Result 3: slit {i+1} has no ionization zone")
            continue
    
        
        # good version of polygon filling using alphashape
        coords = np.vstack([coords_first, coords_last])
    
        if len(coords):
            coords_list.append(coords)       
        else:
            coords_list.append(None)
            print(f"Result 1: slit {i+1} has no ionization zone")
            

# plot sv

slits_orig = n_slit

# Pulling cartesian-plasma_coordinates converter function
T15_R = 1.5 # [m]
rn = 3
log_str = None
volume = 0
log = []

T15_separatrix = []
for sep in geomT15.sep:
    T15_separatrix.append(np.array([sep[0]+T15_R, sep[1]]))

coord_converter = CoordConverterDShapeSimple(T15_R, T15_separatrix)

if slits_orig == 'all':
    slits_orig = 0

slit_number = 0 + slits_orig
for coord in coords_list:
    if coord is not None:
        
    # try:
        hull = ConvexHull(coord)
        volume = ConvexHull(coord).volume * 1e6 # [cm^3] calc volume if iz
        # draw the polygons of the convex hull
        for s in hull.simplices:
            
            color_3 = 'black'
            
            tri = Poly3DCollection([coord[s]])
            tri.set_color(color_3)
            tri.set_alpha(0.2)
            
            axis[0][0].plot(coord[s, 0], coord[s, 1], color=color_3, alpha=0.5)
            axis[0][1].plot(coord[s, 0], coord[s, 2], color=color_3, alpha=0.5)
            axis[1][0].plot(coord[s, 1], coord[s, 2], color=color_3, alpha=0.5)
            
        hull = ConvexHull(coord)
        # draw the polygons of the convex hull
        for s in hull.simplices:
            tri = Poly3DCollection([coord[s]])
            tri.set_color(color_3)
            tri.set_alpha(0.2)
            axis[1][1].add_collection3d(tri)
            
            
            
        label = f'slit {slit_number+1}\nvolume {round(volume, 2)} cm^3'
        # draw the vertices
        
        # axis[1][1].scatter(coord[:, 0], coord[:, 1], coord[:, 2], marker='o', 
        #            color='red', facecolors='white', label=label)
        
        axis[0][0].scatter(coord[:, 0], coord[:, 1], marker='o', 
                   color='red', facecolors='white', label=label)
        
        axis[0][1].scatter(coord[:, 0], coord[:, 2], marker='o', 
                   color='red', facecolors='white', label=label)
        
        axis[1][0].scatter(coord[:, 1], coord[:, 2], marker='o', 
                   color='red', facecolors='white', label=label)
        
        # ionization zone bounds
        # cartesian
        x_min, x_max = min(coord[:, 0])*100, max(coord[:, 0]*100)
        y_min, y_max = min(coord[:, 1])*100, max(coord[:, 1]*100)
        z_min, z_max = min(coord[:, 2])*100, max(coord[:, 2]*100)
        # plasma coordinates
        coord_plasma = []
        for r in coord:
            rho, theta, phi = coord_converter(r)
            coord_plasma.append([rho, theta, phi])
        coord_plasma = np.array(coord_plasma)
        rho_min, rho_max = min(coord_plasma[:, 0]), max(coord_plasma[:, 0])
        theta_min, theta_max = min(coord_plasma[:, 1]), max(coord_plasma[:, 1])
        phi_min, phi_max = min(coord_plasma[:, 2]), max(coord_plasma[:, 2])
        
        log_str = f'slit {slit_number+1}\n' +\
              'ionization zones sizes\n' +\
              f'x, y, z, cm: {round(x_max-x_min, rn)}, ' +\
              f'{round(y_max-y_min, rn)}, ' +\
              f'{round(z_max-z_min, rn)}\n' +\
              f'rho, theta, phi: {round(rho_max-rho_min,rn)}, ' +\
              f'{round(theta_max-theta_min, rn)}, ' +\
              f'{round(phi_max-phi_min, rn)}\n' +\
              f'volume: {round(volume, rn)} cm^3\n'
        log.append(log_str)
        print(log_str)
        
        axis[1][1].set_xlim(x_min/100, x_max/100)
        axis[1][1].set_ylim(y_min/100, y_max/100)
        axis[1][1].set_zlim(z_min/100, z_max/100)
        
        # Set font size
        fontsize_labels = 20
        fontsize_ticks = 15
        labelpad = -5
        pad=5
        labelpad3d=-8
        
        # Set font properties for tick labels
        # ax.xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)  # increase tick size
        # ax.yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        # ax.zaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        
        axis[1][1].set_xlabel('x', **font, fontsize=fontsize_labels, labelpad=labelpad3d)
        axis[1][1].set_ylabel('y', **font, fontsize=fontsize_labels, labelpad=labelpad3d)
        axis[1][1].set_zlabel('z', **font, fontsize=fontsize_labels, labelpad=labelpad3d)
        # Set font properties for tick labels
        axis[1][1].xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)  # increase tick size
        axis[1][1].yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        axis[1][1].zaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        
        axis[0][0].xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)  # increase tick size
        axis[0][0].yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        
        axis[0][1].xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)  # increase tick size
        axis[0][1].yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        
        axis[1][0].xaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)  # increase tick size
        axis[1][0].yaxis.set_tick_params(width=2, **font2, labelsize=fontsize_ticks, pad=pad)
        
        # plt.xticks(ticks=plt.xticks()[0][1:], labels=1000 * np.array(plt.xticks()[0][1:], dtype=np.float64))
        
        axis[0][0].set_xlabel('x', **font, fontsize=fontsize_labels, labelpad=labelpad)
        axis[0][0].set_ylabel('y', **font, fontsize=fontsize_labels, labelpad=labelpad)
        
        axis[0][1].set_xlabel('x', **font, fontsize=fontsize_labels, labelpad=labelpad)
        axis[0][1].set_ylabel('z', **font, fontsize=fontsize_labels, labelpad=labelpad)
        
        axis[1][0].set_xlabel('y', **font, fontsize=fontsize_labels, labelpad=labelpad)
        axis[1][0].set_ylabel('z', **font, fontsize=fontsize_labels, labelpad=labelpad)
        
        # plt.legend()
        plt.show()
        # except:
        #     log.append(f'slit {slit_number+1}\nionization zone is flat (2D)\n')
        #     print('ionization zone is flat (2D)')
    slit_number +=1

plot_lines = False
plot_dots = True
    
def plot_prim(tr, ax, axes='XY', color='k', full_primary=False, alpha=1):
    '''
    plot primary trajectory
    '''
    axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
    # index_X, index_Y = axes_dict[axes]
    index_X, index_Y, index_Z = (0, 1, 2)
    index = -1

    if min(tr.RV_sec.shape) == 0:
        full_primary = True

    if not full_primary:
        # find where secondary trajectory starts:
        for i in range(tr.RV_prim.shape[0]):
            if np.linalg.norm(tr.RV_prim[i, :3] - tr.RV_sec[0, :3]) < 1e-4:
                index = i+1
                
    axis[1][1].plot(tr.RV_prim[:index, index_X],
            tr.RV_prim[:index, index_Y],
            tr.RV_prim[:index, index_Z],
            color=color, linewidth=2, alpha=alpha)
    
for tr in traj_fat_beam:
    if plot_lines:
        # plot primary trajectory
        plot_prim(tr, axis[1][1], axes='XY', color='k', full_primary=True, alpha=0.5)
    
    # plot secondaries
    for i in slits:
        c = colors[i]
        if tr.RV_sec_toslits[i] is None:
            continue
        for fan_tr in tr.RV_sec_toslits[i]:
            if plot_lines:
                axis[1][1].plot(fan_tr[:, 0], fan_tr[:, 1], fan_tr[:, 2], color=c, alpha=0.5)
                
            if plot_dots:
                # plot zones
                
                # axis[1][1].plot(fan_tr[0, 0], fan_tr[0, 1], fan_tr[0, 2], 'o', color=c,
                #           markerfacecolor='white', alpha=0.5)
                
                axis[0][0].plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                          markerfacecolor='white', alpha=0.5)
                
                axis[0][1].plot(fan_tr[0, 0], fan_tr[0, 2], 'o', color=c,
                          markerfacecolor='white', alpha=0.5)
                
                axis[1][0].plot(fan_tr[0, 1], fan_tr[0, 2], 'o', color=c,
                          markerfacecolor='white', alpha=0.5)

axis[0][0].xaxis.set_ticklabels([])
axis[0][0].yaxis.set_ticklabels([])

axis[0][1].xaxis.set_ticklabels([])
axis[0][1].yaxis.set_ticklabels([])

axis[1][0].xaxis.set_ticklabels([])
axis[1][0].yaxis.set_ticklabels([])

axis[1][1].xaxis.set_ticklabels([])
axis[1][1].yaxis.set_ticklabels([])
axis[1][1].zaxis.set_ticklabels([])

axis[1][1].view_init(elev=27., azim=125., roll=0.)

axis[0][0].grid(which='major', color='tab:gray')  # draw primary grid
axis[0][0].minorticks_on()  # make secondary ticks on axes
# draw secondary grid
axis[0][0].grid(which='minor', color='tab:gray', linestyle=':')


axis[0][1].grid(which='major', color='tab:gray')  # draw primary grid
axis[0][1].minorticks_on()  # make secondary ticks on axes
# draw secondary grid
axis[0][1].grid(which='minor', color='tab:gray', linestyle=':')

axis[1][0].grid(which='major', color='tab:gray')  # draw primary grid
axis[1][0].minorticks_on()  # make secondary ticks on axes
# draw secondary grid
axis[1][0].grid(which='minor', color='tab:gray', linestyle=':')


fig_sv3.subplots_adjust(left=0.25, bottom=0.05, right=0.80, top=0.95)

fig_sv3.subplots_adjust(wspace=0.1, hspace=0.1)
