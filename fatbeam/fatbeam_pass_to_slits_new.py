# %% imports
import os
import copy
import math
import errno
import numpy as np
import pandas as pd
import pickle as pc

from itertools import cycle
from matplotlib import path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed # joblib=1.3.2; python>=3.11.6;
from matplotlib.patches import Rectangle

import hibpplotlib as hbplot
import optimizers
import hibpcalc.geomfunc as gf
import hibplib as hb

#from hibpcalc.fields import return_E
from hibpcalc.fields import __return_E as return_E
from hibpcalc.misc import runge_kutt, argfind_rv, find_fork
from fatbeam.fatbeam_grid import fatbeam_set_new_RV0s2

#%%

def pass_to_slits(tr, dt, E, B, geom, target='slit', timestep_divider=10,
                  slits=range(7), no_intersect=True, no_out_of_bounds=True,
                  print_log=True):
    '''
    pass trajectories to slits and save secondaries which get into slits
    '''
    # find the number of slits
    n_slits = geom.plates_dict['an'].slits_edges.shape[0]
    tr.add_slits(n_slits)
    
    # find slits position
    if target == 'slit':
        r_slits = geom.plates_dict['an'].slits_edges
        rs = geom.r_dict['slit']
        slit_plane_n = geom.plates_dict['an'].slit_plane_n
        slits_spot = geom.plates_dict['an'].slits_spot
    elif target == 'det':
        r_slits = geom.plates_dict['an'].det_edges
        rs = geom.r_dict['det']
        slit_plane_n = geom.plates_dict['an'].det_plane_n
        slits_spot = geom.plates_dict['an'].det_spot

    # create slits polygon
    ax_index = np.argmax(slit_plane_n)
    slits_spot_flat = np.delete(slits_spot, ax_index, 1)
    slits_spot_poly = path.Path(slits_spot_flat)

    # find index of primary trajectory point where secondary starts
    index = np.nanargmin(np.linalg.norm(tr.RV_prim[:, :3] -
                                        tr.RV_sec[0, :3], axis=1))
    sec_ind = range(index-2, index+2)
    
    if print_log:
        print('\nStarting precise fan calculation')
        
    # divide the timestep
    tr.dt1 = dt/timestep_divider
    tr.dt2 = dt
    k = tr.q / tr.m
    
    # number of steps during new fan calculation
    n_steps = timestep_divider * (len(sec_ind))
    
    # list for new trajectories
    fan_list = []
    
    # take the point to start fan calculation
    RV_old = tr.RV_prim[sec_ind[0]]
    RV_old = np.array([RV_old])
    RV_new = RV_old

    i_steps = 0
    inside_slits_poly = False
    while i_steps <= n_steps:
        # pass new secondary trajectory
        tr.pass_sec(RV_new, rs, E, B, geom,
                    stop_plane_n=slit_plane_n, tmax=9e-5,
                    eps_xy=1e-3, eps_z=1, print_log=False)

        # make a step on primary trajectory
        r = RV_old[0, :3]

        # fields
        E_local = np.array([0., 0., 0.])
        B_local = B(r)
        if np.isnan(B_local).any():
            if print_log:
                print('Btor is nan, r = %s' % str(r))
            break

        # runge-kutta step
        RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
        RV_old = RV_new
        i_steps += 1

        # check intersection with slits polygon
        intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
        contains_point = slits_spot_poly.contains_point(intersect_coords_flat)

        # save result
        if not (True in tr.IntersectGeometrySec.values() or
                tr.B_out_of_bounds) and contains_point:
            inside_slits_poly = True
            fan_list.append(tr.RV_sec)
        if not contains_point and inside_slits_poly:
            break
    if print_log:
        print('\nPrecise fan calculated')

    # choose secondaries which get into slits
    # start slit cycle
    for i_slit in slits:
        if print_log:
            print('\nslit = {}'.format(i_slit+1))
            print('center of the slit = ', r_slits[i_slit, 0, :], '\n')

        # create slit polygon
        slit_flat = np.delete(r_slits[i_slit, 1:, :], ax_index, 1)
        slit_poly = path.Path(slit_flat)
        zones_list = []  # list for ion zones coordinates
        rv_list = []  # list for RV arrays of secondaries

        for fan_tr in fan_list:
            # get last coordinates of the secondary trajectory
            intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
            if slit_poly.contains_point(intersect_coords_flat):
                if print_log:
                    print('slit {} ok!\n'.format(i_slit+1))
                rv_list.append(fan_tr)
                zones_list.append(fan_tr[0, :3])

        tr.RV_sec_toslits[i_slit] = rv_list
        tr.ion_zones[i_slit] = np.array(zones_list)
    tr.fan_to_slits = fan_list

    return tr