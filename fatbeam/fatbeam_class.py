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

#local imports
import hibplib as hb
import hibpcalc as hc
import hibpcalc.geomfunc as gf

from hibpcalc.fields import __return_E as return_E
from hibpcalc.misc import runge_kutt, argfind_rv, find_fork

class Fatbeam: 
    
    traj: hb.Traj # initial trajectory
    filaments: list[hb.Traj] # list of filaments. Filament: hb.Traj
    E: dict
    B: hc.fields.FieldInterpolator
    geom: hb.Geometry
    Btor: float
    Ipl: float
    
    def __init__(self, traj: hb.Traj, E: dict, 
                 B: hc.fields.FieldInterpolator, geom: hb.Geometry,
                Btor: float, Ipl: float) -> None:

        self.traj = traj
        self.filaments = []
        self.E = E
        self.B = B
        self.geom = geom
        self.Btor = Btor
        self.Ipl = Ipl
        self.grid = []
        self.d_beam = 0.02
        
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
            new_tr.dt1 = 0.7e-7 # dt for passing primary trajectory
            new_tr.dt2 = 0.7e-7 # dt for passing secondary trajectories
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
    def pass_to_slits(tr, dt, E, B, geom, target='slit', timestep_divider=1,
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
        sec_ind = tr.RV_prim #range(index-2, index+2)
        # sec_ind = len(tr.RV_prim)
        
        if print_log:
            print('\nStarting precise fan calculation')
            
        # divide the timestep
        tr.dt1 = dt/timestep_divider
        tr.dt2 = dt
        k = tr.q / tr.m
        
        # number of steps during new fan calculation
        n_steps = timestep_divider * len(sec_ind)
        
        # list for new trajectories
        fan_list = []
        
        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        mask = np.sqrt((tr.RV_prim[:, 0] - geom.R)**2 +
                       (tr.RV_prim[:, 1] / geom.elon)**2) <= geom.r_plasma
        
        # take the point to start fan calculation
        # RV_old = tr.RV_prim[sec_ind[0]]
        RV_old = tr.RV_prim[0]
        RV_old = np.array([RV_old])
        RV_new = RV_old
        
        inside_slits_poly = False
        for i_steps in tqdm(range(1, n_steps)):
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
            # RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
            RV_new = np.array([tr.RV_prim[i_steps]])
            RV_old = RV_new
    
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
        
        # i_steps = 1
        # inside_slits_poly = False
        # while i_steps <= n_steps:
        #     # pass new secondary trajectory
        #     tr.pass_sec(RV_new, rs, E, B, geom,
        #                 stop_plane_n=slit_plane_n, tmax=9e-5,
        #                 eps_xy=1e-3, eps_z=1, print_log=False)
    
        #     # make a step on primary trajectory
        #     r = RV_old[0, :3]
            
        #     # fields
        #     E_local = np.array([0., 0., 0.])
        #     B_local = B(r)
        #     if np.isnan(B_local).any():
        #         if print_log:
        #             print('Btor is nan, r = %s' % str(r))
        #         break
    
        #     # runge-kutta step
        #     # RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
        #     RV_new = np.array([tr.RV_prim[i_steps]])
        #     RV_old = RV_new
        #     i_steps += 1
    
        #     # check intersection with slits polygon
        #     intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
        #     contains_point = slits_spot_poly.contains_point(intersect_coords_flat)
    
        #     # save result
        #     if not (True in tr.IntersectGeometrySec.values() or
        #             tr.B_out_of_bounds) and contains_point:
        #         inside_slits_poly = True
        #         fan_list.append(tr.RV_sec)
        #     if not contains_point and inside_slits_poly:
        #         break
        # if print_log:
        #     print('\nPrecise fan calculated')
    
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