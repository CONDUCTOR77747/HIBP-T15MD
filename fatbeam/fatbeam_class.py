# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:58:35 2023

@author: ammosov_yam
"""

import os
import copy
import numpy as np
from tqdm import tqdm
from matplotlib import path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed # joblib=1.3.2; python>=3.11.6;

#local imports
import hibplib as hb
import hibpcalc as hc
import hibpcalc.geomfunc as gf

from hibpcalc.fields import __return_E as return_E
from hibpcalc.misc import runge_kutt, argfind_rv, find_fork

def pass_sec_parallel(tr, prim_point, rs, E, B, geom, slit_plane_n,
ax_index, slits_spot_flat, slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, 
any_traj=False, print_log=False):
    #create an instance of trajectory with only 1 RV_prim point
    # tr = hb.Traj(old_tr.q, old_tr.m, old_tr.Ebeam, prim_point, old_tr.alpha, 
    #              old_tr.beta, old_tr.U)
    # tr.dt1 = old_tr.dt1 # dt for passing primary trajectory
    # tr.dt2 = old_tr.dt2 # dt for passing secondary trajectories
    # tr.I0 = old_tr.I0 # set I0 initial current distribution       
    
    tr.RV_prim = np.array([prim_point])
    
    # # take the point to start fan calculation
    # RV_old = tr.RV_prim[0]
    # RV_old = np.array([RV_old])
    # RV_new = RV_old
    
    # # inside_slits_poly = False
    
    # # make a step on primary trajectory
    # r = RV_old[:3]
    
    # # fields
    # E_local = np.array([0., 0., 0.])
    # B_local = B(r)
    # if np.isnan(B_local).any():
    #     if print_log:
    #         print(f'Btor is nan, r = {r}')
    #     return None
    
    # pass new secondary trajectory
    tr.pass_sec(tr.RV_prim, rs, E, B, geom,
                stop_plane_n=slit_plane_n, tmax=9e-5,
                eps_xy=1e-3, eps_z=1, print_log=False)
    
    # check intersection with slits polygon
    intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
    contains_point = slits_spot_poly.contains_point(intersect_coords_flat)

    # save result
    if (not (True in tr.IntersectGeometrySec.values() or
            tr.B_out_of_bounds) and contains_point) or any_traj:
        # inside_slits_poly = True
        return tr.RV_sec
    # if not contains_point and inside_slits_poly:
    #     return None

def find_diapason(tr, fan_list, intersect_geom_list, step, rs, E, B, geom, 
                  slit_plane_n, ax_index, slits_spot_flat, slits_spot_poly,
                  s_s_s = None, any_traj=False, print_log=False):
    
    # choose secondaries which bump into geometry from intersect_geom_list
    precise_diapason_list = []
    if intersect_geom_list:
        for traj in fan_list:
            intersected = False
            
            for i in range(len(traj)-1):
                if intersected:
                    break
                point1 = traj[i, :3]
                point2 = traj[i+1, :3]
                    
                for intersect_geom in intersect_geom_list:
                    if intersect_geom.check_intersect(point1, point2):
                        precise_diapason_list.append(traj)
                        intersected = True
                        break
    
    # setup start and stop index (and step) of primary traj. for passing secondaries
    if s_s_s is None:
        start = np.where(tr.RV_prim==precise_diapason_list[0][0])[0][0]
        stop = np.where(tr.RV_prim==precise_diapason_list[-1][0])[0][0]
        step = step
        masked_list = [tr.RV_prim[i] for i in range(start, stop, step)]
    else:
        start = s_s_s[0]
        stop = s_s_s[1]
        step = s_s_s[2]
        if stop == -1:
            stop = len(tr.RV_prim)
            masked_list = [tr.RV_prim[i] for i in range(start, stop, step)]
        else:
            masked_list = [tr.RV_prim[i] for i in range(start, stop, step)]
        
    if print_log:   
        print(f'Primary points: {len(masked_list)}, dt1: {tr.dt1}, dt2: {tr.dt2}')
    
    # main calc multiprocessing using joblib=1.3.2
    # pass secondary traj. in parallel
    fan_list = Parallel(n_jobs=-1, verbose=5)(delayed(pass_sec_parallel)(tr,
    primary_point, rs, E, B, geom, slit_plane_n, ax_index, slits_spot_flat,
    slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, any_traj=any_traj, 
    print_log=False) for primary_point in masked_list)
    
    if print_log:
        print('\nPrecise fan calculated')
        
    return fan_list, precise_diapason_list


class Fatbeam: 
    
    traj: hb.Traj # initial trajectory
    filaments: list[hb.Traj] # list of filaments. Filament: hb.Traj
    E: dict
    B: hc.fields.FieldInterpolator
    geom: hb.Geometry
    Btor: float
    Ipl: float
    dt1: float
    dt2: float
    
    def __init__(self, traj: hb.Traj, E: dict, 
                 B: hc.fields.FieldInterpolator, geom: hb.Geometry,
                Btor: float, Ipl: float, dt1:float=3e-10, dt2:float=3e-10) -> None:

        self.traj = traj
        self.filaments = []
        self.E = E
        self.B = B
        self.geom = geom
        self.Btor = Btor
        self.Ipl = Ipl
        self.grid = []
        self.d_beam = 0.02
        self.dt1 = dt1
        self.dt2 = dt2
        
    def calc(self,  d_beam, foc_len, dt=2e-8, filaments=7, slits=[3], target='slits'):
        timestep_divider=20
        pass
    
    def save(self, dirname: str) -> None:
        pass
    
    def plot(self):
        pass
    
    def _gauss2d(self, x=0, y=0, mx=0, my=0, sx=0.003, sy=0.003, a=1, pad=0):
        return a * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.))) + pad
    
    def create_grid(self, c0, r, n, profile):
        """
        
        Creates 2D rectangle grid of points inside circle with raidus r.

        Parameters
        ----------
        c0 : list of float
            Circle central dot coordinates [m]. Example: c0 = [0, 0].
        r : float
            Radius of a circle [m].
        n : int
            Amount of dots along vertical and horizontal diameters of rectangle grid.
            Odd values are preferred.
        profile : function
            Function with given distribution profile.    
            Gauss function by default.
            
        Returns
        -------
        grid : list of three lists of floats - [[x values], [y values], [z values]]
            list with x, y values of grid and z values of profile function.
            Example: X and Y in meters, Z - values of Gauss function at [X, Y] point.

        """
        
        c = [0, 0] # create grid for [0, 0] point and then add c0 values

        # only 1 central dot
        if n == 1:
            return [[c[0]], [c[1]], [0]]

        grid = [[],[],[]]
        
        # set grid boundary points
        left_point  = c[0] - r
        right_point = c[0] + r
        down_point  = c[1] - r
        up_point    = c[1] + r
        
        # create horizontal and vertical diameters of rectangle grid
        x = np.linspace(left_point, right_point, int(n))
        y = np.linspace(down_point, up_point, int(n))
        
        # create rectangle grid of points
        X, Y = np.meshgrid(x, y)
        # Compute the profile function at each point
        sx=sy=0.003
        Z = profile(X, Y, c[0], c[1], sx, sy)
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # take points of rectangle grid, which inside the circle with radius r
        for i in range(len(positions[0])):
            p = [positions[0][i], positions[1][i], positions[2][i]]
            if (p[0]-c[0])**2 + (p[1]-c[1])**2 <= r**2:
                grid[0].append(p[0]+c0[0]) # Z axis
                grid[1].append(p[1]+c0[1]) # Y axis
                grid[2].append(p[2]) # Z axis - profile func values
        
        self.grid = grid
        return grid
    
    def _set_new_RV0s(self, d_beam, foc_len, n=7):
        
        """

        Parameters
        ----------
        tr : hb.Traj
            Optimized single Trajectory.
        d_beam : float
            beam diameter [m].
        foc_len : float
            focal length of the beam [m].
        Grid : Class
            Class for creation grid. For example Rectangular Gauss Grid.
        n : int
            Amount of dots along vertical and horizontal diameters of rectangle grid.

        Returns
        -------
        tr_fat_new_rv0 : list[hb.Traj]
            List of Trajectries with new RV0 setted according to grid.

        """
        grid = self.create_grid([0, 0], d_beam/2., n, self._gauss2d)
        
        tr = self.traj # initial trajectory (thin, one E, one U)
        tr_fat_new_rv0 = self.filaments # list contains trajectories with new RV0
        
        r0 = tr.RV0[0, :3] # starting position coordinates
        v0_abs = np.linalg.norm(tr.RV0[0, 3:]) # absolute velocity
        
        n_filaments = len(grid[0])
        for i in range(n_filaments):

            # beam convergence angle
            if n != 1:
                alpha_conv = np.arctan((i - (n_filaments-1)/2) *
                                    (d_beam/(n_filaments-1)) / foc_len)
            elif n == 1:
                alpha_conv = 0
            
            
            r = [grid[0][i], grid[1][i], 0]
            
            v0 = v0_abs * np.array([-np.cos(alpha_conv),
                                np.sin(alpha_conv), 0.])
           
            # create filaments new starting positions by rotating and shifting
            r_rot = gf.rotate(np.array(r), axis=(1, 0, 0), deg=90)
            r_rot = gf.rotate(r_rot, axis=(0, 0, 1), deg=tr.alpha+90)
            r_rot = gf.rotate(r_rot, axis=(0, 1, 0), deg=tr.beta)
            r_rot += r0
            v_rot = gf.rotate(v0, axis=(1, 0, 0), deg=90)
            v_rot = gf.rotate(v_rot, axis=(0, 0, 1), deg=tr.alpha)
            v_rot = gf.rotate(v_rot, axis=(0, 1, 0), deg=tr.beta)
            
            new_tr = hb.Traj(tr.q, tr.m, tr.Ebeam, r0, tr.alpha, tr.beta, tr.U)
            new_tr.dt1 = self.dt1 # dt for passing primary trajectory
            new_tr.dt2 = self.dt2 # dt for passing secondary trajectories
            new_tr.I0 = grid[2][i] # set I0 initial current distribution
            new_tr.RV0[0, :] = np.hstack([r_rot, v_rot])
            tr_fat_new_rv0.append(new_tr)

    def plot3d(self):
    
        c = [0, 0]
        r = self.d_beam/2.
        grid = self.grid
    
        # Plot the 3D surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(grid[0], grid[1], grid[2])
        ax.plot_trisurf(grid[0], grid[1], grid[2], linewidth=1, alpha=0.5, color='green')
        
        # ax.scatter(grid[0], grid[1], color='blue')
        ax.scatter(grid[0], grid[1], grid[2], color='red')
        
        # plot radius - red circle
        cx, cy = [], [] 
        for i in range(0, 51):
          t = 2.0 * np.pi * i / 50.0
          cx.append(r * np.cos(t) + c[0])
          cy.append(r * np.sin(t) + c[1])
        ax.plot(cx, cy, linewidth=2.0, color='blue' )
        
        # Set labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()     
    
    
    
    @staticmethod
    def pass_to_slits_parallel_plus(tr, E, B, geom, target='slit', step_pass_sec=1,
                      slits=range(7), any_traj=False, print_log=True):
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
        
        if print_log:
            print('\nStarting precise fan calculation')
        
        # number of steps during new fan calculation
        n_steps = len(tr.RV_prim)
        
        # list for new trajectories
        fan_list = []

        # geometry mask
        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        masked_list_0 = []
        for i in range(len(tr.RV_prim)):
            if np.sqrt((tr.RV_prim[i, 0] - geom.R)**2 + 
                       (tr.RV_prim[i, 1] / geom.elon)**2) > geom.r_plasma:
                # masked_list.append(None)
                continue
            masked_list_0.append(tr.RV_prim[i])
        
        # first step. along ALL primary traj.
        masked_list = []
        start = 0
        stop = len(masked_list_0)
        step = step_pass_sec
        masked_list = [masked_list_0[i] for i in range(start, stop, step)]
            
        if print_log:
            print(f'Primary points: {len(masked_list)}, dt1: {tr.dt1}, dt2: {tr.dt2}')
        
        # main calc multiprocessing using joblib=1.3.2
        fan_list = Parallel(n_jobs=-1, verbose=5)(delayed(pass_sec_parallel)(tr,
        primary_point, rs, E, B, geom, slit_plane_n, ax_index, slits_spot_flat,
        slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, any_traj=any_traj, 
        print_log=False) for primary_point in masked_list)
        
        # second step. between A3 plates
        intersect_geom_list = [geom.plates_dict['A3'], geom.plates_dict['A3d']]
        step = 100
        fan_list, precise_diapason_list = find_diapason(tr, fan_list, 
                        intersect_geom_list, step, rs, E, B, geom, 
                          slit_plane_n, ax_index, slits_spot_flat, slits_spot_poly, 
                          s_s_s=None, any_traj=any_traj, print_log=print_log)
        
        # thrid step. between A4 plates
        intersect_geom_list = [geom.plates_dict['A4'], geom.plates_dict['A4d']]
        step = 100
        fan_list, precise_diapason_list = find_diapason(tr, fan_list, 
                        intersect_geom_list, step, rs, E, B, geom, 
                          slit_plane_n, ax_index, slits_spot_flat, slits_spot_poly, 
                          s_s_s=None, any_traj=any_traj, print_log=print_log)
        
        # fourth step. between ENTIRE slit plate
        # intersect_geom_list = [geom.plates_dict['an']]
        # step = 100
        # fan_list, precise_diapason_list = find_diapason(tr, fan_list, 
        #                 intersect_geom_list, step, rs, E, B, geom, 
        #                   slit_plane_n, ax_index, slits_spot_flat, slits_spot_poly, 
        #                   s_s_s=None, any_traj=any_traj, print_log=print_log)
        
        # # check intersection with slits polygon
        # intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
        # contains_point = slits_spot_poly.contains_point(intersect_coords_flat)

        # # save result
        # if (not (True in tr.IntersectGeometrySec.values() or
        #         tr.B_out_of_bounds) and contains_point) or any_traj:
        #     inside_slits_poly = True
        #     fan_list.append(tr.RV_sec)
        # # if not contains_point and inside_slits_poly:
        # #     break
        
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
                try:
                    # get last coordinates of the secondary trajectory
                    intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
                    if slit_poly.contains_point(intersect_coords_flat) or any_traj:
                        if print_log:
                            print('slit {} ok!\n'.format(i_slit+1))
                        rv_list.append(fan_tr)
                        zones_list.append(fan_tr[0, :3])
                except:
                    continue
    
            tr.RV_sec_toslits[i_slit] = rv_list
            tr.ion_zones[i_slit] = np.array(zones_list)
        tr.fan_to_slits = fan_list
    
        return tr
    
    @staticmethod
    def pass_to_slits_parallel_plus_old(tr, E, B, geom, target='slit', step_pass_sec=1,
                      slits=range(7), any_traj=False, print_log=True):
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
        
        if print_log:
            print('\nStarting precise fan calculation')
        
        # number of steps during new fan calculation
        n_steps = len(tr.RV_prim)
        
        # list for new trajectories
        fan_list = []

        # geometry mask
        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        masked_list_0 = []
        for i in range(len(tr.RV_prim)):
            if np.sqrt((tr.RV_prim[i, 0] - geom.R)**2 + 
                       (tr.RV_prim[i, 1] / geom.elon)**2) > geom.r_plasma:
                # masked_list.append(None)
                continue
            masked_list_0.append(tr.RV_prim[i])
        
        masked_list = []
        start = 0
        stop = len(masked_list_0)
        step = step_pass_sec
        masked_list = [masked_list_0[i] for i in range(start, stop, step)]
            
        if print_log:
            print(f'Primary points: {len(masked_list)}, dt1: {tr.dt1}, dt2: {tr.dt2}')
        
        # main calc multiprocessing using joblib=1.3.2
        fan_list = Parallel(n_jobs=-1, verbose=5)(delayed(pass_sec_parallel)(tr,
        primary_point, rs, E, B, geom, slit_plane_n, ax_index, slits_spot_flat,
        slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, any_traj=any_traj, 
        print_log=False) for primary_point in masked_list)
        
        if print_log:
            print('\nPrecise fan calculated')
        
        precise_diapason_list = []
        # choose secondaries which bump into upper A3 and lower plate A3d (A3 plates)
        for traj in fan_list:
            # print(f'Check traj: {traj}')
            for i in range(len(traj)-1):
                point1 = traj[i, :3]
                point2 = traj[i+1, :3]
                if (geom.plates_dict['A3'].check_intersect(point1, point2) or
                geom.plates_dict['A3d'].check_intersect(point1, point2)):
                    precise_diapason_list.append(traj)
                    break
        
        # fan_list = [precise_diapason_list[0], precise_diapason_list[-1]]
        
        # fan_list = precise_diapason_list
        
        masked_list = []
        start = np.where(tr.RV_prim==precise_diapason_list[0][0])[0][0]
        stop = np.where(tr.RV_prim==precise_diapason_list[-1][0])[0][0]
        step = 100
        
        masked_list = [tr.RV_prim[i] for i in range(start, stop, step)]
        
        if print_log:   
            print(f'Primary points 2: {len(masked_list)}, dt1: {tr.dt1}, dt2: {tr.dt2}')
        
        # main calc multiprocessing using joblib=1.3.2
        fan_list = Parallel(n_jobs=-1, verbose=5)(delayed(pass_sec_parallel)(tr,
        primary_point, rs, E, B, geom, slit_plane_n, ax_index, slits_spot_flat,
        slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, any_traj=any_traj, 
        print_log=False) for primary_point in masked_list)
        
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
                try:
                    # get last coordinates of the secondary trajectory
                    intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
                    if slit_poly.contains_point(intersect_coords_flat) or any_traj:
                        if print_log:
                            print('slit {} ok!\n'.format(i_slit+1))
                        rv_list.append(fan_tr)
                        zones_list.append(fan_tr[0, :3])
                except:
                    continue
    
            tr.RV_sec_toslits[i_slit] = rv_list
            tr.ion_zones[i_slit] = np.array(zones_list)
        tr.fan_to_slits = fan_list
    
        return tr
    
    @staticmethod
    def pass_to_slits_parallel(tr, E, B, geom, target='slit', step_pass_sec=1,
                      slits=range(7), any_traj=False, print_log=True):
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
        
        if print_log:
            print('\nStarting precise fan calculation')
        
        # number of steps during new fan calculation
        n_steps = len(tr.RV_prim)
        
        # list for new trajectories
        fan_list = []

        # geometry mask
        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        masked_list_0 = []
        for i in range(len(tr.RV_prim)):
            if np.sqrt((tr.RV_prim[i, 0] - geom.R)**2 + 
                       (tr.RV_prim[i, 1] / geom.elon)**2) > geom.r_plasma:
                # masked_list.append(None)
                continue
            masked_list_0.append(tr.RV_prim[i])
        
        masked_list = []
        start = 0
        stop = len(masked_list_0)
        step = step_pass_sec
        masked_list = [masked_list_0[i] for i in range(start, stop, step)]
            
        print(f'Primary points: {len(masked_list)}, dt1: {tr.dt1}, dt2: {tr.dt2}')
        
        # main calc multiprocessing using joblib=1.3.2
        fan_list = Parallel(n_jobs=-1, verbose=5)(delayed(pass_sec_parallel)(tr,
        primary_point, rs, E, B, geom, slit_plane_n, ax_index, slits_spot_flat,
        slits_spot_poly, tmax=9e-5, eps_xy=1e-3, eps_z=1, any_traj=any_traj, 
        print_log=False) for primary_point in masked_list)
        
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
                try:
                    # get last coordinates of the secondary trajectory
                    intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
                    if slit_poly.contains_point(intersect_coords_flat) or any_traj:
                        if print_log:
                            print('slit {} ok!\n'.format(i_slit+1))
                        rv_list.append(fan_tr)
                        zones_list.append(fan_tr[0, :3])
                except:
                    continue
    
            tr.RV_sec_toslits[i_slit] = rv_list
            tr.ion_zones[i_slit] = np.array(zones_list)
        tr.fan_to_slits = fan_list
    
        return tr
    
    @staticmethod
    def pass_to_slits(tr, E, B, geom, target='slit', step_on_primary=1,
                      slits=range(7), any_traj=False, print_log=True):
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
        # index = np.nanargmin(np.linalg.norm(tr.RV_prim[:, :3] -
        #                                     tr.RV_sec[0, :3], axis=1))
        sec_ind = tr.RV_prim #range(index-2, index+2) #
        
        if print_log:
            print('\nStarting precise fan calculation')
            
        # divide the timestep
        # tr.dt1 = dt
        # tr.dt2 = dt
        # k = tr.q / tr.m
        
        # for i in range(len(tr.RV_prim)):
        #     print("check", tr.RV_prim[i, 0], tr.RV_prim[i, 1])
        
        # number of steps during new fan calculation
        n_steps = len(sec_ind)
        
        # list for new trajectories
        fan_list = []
        
        # take the point to start fan calculation
        # RV_old = tr.RV_prim[sec_ind[0]]
        RV_old = tr.RV_prim[0]
        RV_old = np.array([RV_old])
        RV_new = RV_old
        
        inside_slits_poly = False
        for i_steps in tqdm(range(0, n_steps, step_on_primary)):
            
            # geometry mask
            # check eliptical radius of particle:
            # 1.5 m - major radius of a torus, elon - size along Y
            if np.sqrt((tr.RV_prim[i_steps, 0] - geom.R)**2 + 
                        (tr.RV_prim[i_steps, 1] / geom.elon)**2) > geom.r_plasma:
                # print(tr.RV_prim[i_steps, 0], tr.RV_prim[i_steps, 1], 'skipped', f'i={i_steps}')
                continue
            
            # make a step on primary trajectory
            r = RV_old[0, :3]
            
            # fields
            E_local = np.array([0., 0., 0.])
            B_local = B(r)
            if np.isnan(B_local).any():
                if print_log:
                    print(f'Btor is nan, r = {r}')
                break
    
            # runge-kutta step
            # RV_new2 = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
            RV_new = np.array([tr.RV_prim[i_steps]])
            # print("compare", RV_new, RV_new2, sep='\n')
            RV_old = RV_new
            
            
            # pass new secondary trajectory
            tr.pass_sec(RV_new, rs, E, B, geom,
                        stop_plane_n=slit_plane_n, tmax=9e-5,
                        eps_xy=1e-3, eps_z=1, print_log=False)
            
    
            # check intersection with slits polygon
            intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
            contains_point = slits_spot_poly.contains_point(intersect_coords_flat)
    
            # save result
            if (not (True in tr.IntersectGeometrySec.values() or
                    tr.B_out_of_bounds) and contains_point) or any_traj:
                inside_slits_poly = True
                fan_list.append(tr.RV_sec)
            # if not contains_point and inside_slits_poly:
            #     break
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
                if slit_poly.contains_point(intersect_coords_flat) or any_traj:
                    if print_log:
                        print('slit {} ok!\n'.format(i_slit+1))
                    rv_list.append(fan_tr)
                    zones_list.append(fan_tr[0, :3])
    
            tr.RV_sec_toslits[i_slit] = rv_list
            tr.ion_zones[i_slit] = np.array(zones_list)
        tr.fan_to_slits = fan_list
    
        return tr



 # fatbeam_kwargs = {'Ebeam_orig':'260',
 #                   'UA2_orig':'10',
 #                   'target':'slit',
 #                   'slits_orig':'4',
 #                   'd_beam':0.02,
 #                   'foc_len':50,
 #                   'n_filaments':5,
 #                   'n_gamma':5,
 #                   'timestep_divider':20,
 #                   'dt':2e-8,
 #                   'calc_mode':'cpu', # cpu_unparallel
 #                   'load_traj':load_traj,
 #                   'save_traj':save_traj,
 #                   'path_orig': os.path.join('fatbeam', 'results', 'fix_cylinder'),
 #                   'plot_trajs': plot_trajs,
 #                   'rescale_plots': rescale_plots,
 #                   'close_plots': close_plots,
 #                   'save_plots': save_plots,
 #                   'create_table': False}

if __name__=='__main__':
    
    fb = Fatbeam(traj_list_optimized[0], E, B, geomT15, Btor, Ipl)
    fb.calc()
    fb.save(os.path.join('fatbeam', 'results', 'new'))