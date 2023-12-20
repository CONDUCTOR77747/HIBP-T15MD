# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:58:35 2023

@author: ammosov_yam
"""

import hibplib
import hibpcalc
import numpy as np

class fatbeam(): 
    
    fatbeam: hibplib.Traj
    E: dict
    B: hibpcalc.fields.FieldInterpolator
    geom: hibplib.Geometry
    Btor: float
    Ipl: float
    
    
    
    def __init__(self, traj: hibplib.Traj, E: dict, 
                 B: hibpcalc.fields.FieldInterpolator, geom: hibplib.Geometry,
                Btor: float, Ipl: float) -> None:
        self.fatbeam = traj
        self.E = E
        self.B = B
        self.geom = geom
        self.Btor = Btor
        self.Ipl = Ipl
        
    def calc(self,  d_beam, foc_len, dt=2e-8, filaments=10, slits=[3], target='slits'):
        timestep_divider=20
    
    def save(self, dirname):
        pass
    
    def plot(self):
        pass
    
    
    
    def _set_new_RV0s(self, tr, d_beam, foc_len, filaments):
        tr_rots = []
        tr_fat_buff_list = [] # list contains trajectories with new RV0
        r0 = tr.RV0[0, :3]
        v0_abs = np.linalg.norm(tr.RV0[0, 3:])
        n = 7 # Amount of dots along vertical and horizontal diameters of rectangle grid.
        
        grid = create_disk_grid_gauss([0, 0], d_beam/2., n, gaus2d)
        
        n_filaments_xy = len(grid[0])
        
        for i in range(n_filaments_xy):
        # skip center traj
            # if abs(i - (n_filaments_xy-1)/2) < 1e-6 and skip_center_traj:
            #     continue
            
            # beam convergence angle
            alpha_conv = np.arctan((i - (n_filaments_xy-1)/2) *
                                (d_beam/(n_filaments_xy-1)) / foc_len)
            # alpha_conv = 0
            
            # set coords and velocity at the center of coord system
            # x = 0.0
            # y = -(i - (n_filaments_xy-1)/2) * (d_beam/(n_filaments_xy-1))
            # z = 0.0
            
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
            
            tr_fat = copy.deepcopy(tr)
            tr_fat.I0 = grid[2][i]
            tr_fat.RV0[0, :] = np.hstack([r_rot, v_rot])
            tr_rots.append(r_rot)
            tr_fat_buff_list.append(tr_fat)
            
            # if abs(y) < 1e-6:
            #     break
            
        return tr_fat_buff_list, tr_rots

 # fatbeam_kwargs = {'Ebeam_orig':'260',
 #                   'UA2_orig':'10',
 #                   'target':'slit',
 #                   'slits_orig':'4',
 #                   'd_beam':0.02,
 #                   'foc_len':50,
 #                   'n_filaments_xy':5,
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

class grid():
    
    def _create_disk_grid(self, c, r, n):
        """
        
        Creates 2D rectangle grid of points inside circle with raidus r.

        Parameters
        ----------
        c : list of float
            Circle central dot. Example: c = [0, 0].
        r : float
            Radius of a circle.
        n : int
            Amount of dots along vertical and horizontal diameters of rectangle grid.
            Odd values are preferred.

        Returns
        -------
        grid : list of two lists of floats - [[x values], [y values]]
            list with x and y values of grid.

        """
        
        # only 1 central dot
        if n == 1:
            return [[c[0]], [c[1]]]
        
        grid = [[],[]]
        
        # set grid boundary points
        left_point  = c[0] - r
        right_point = c[0] + r
        down_point  = c[1] - r
        up_point    = c[1] + r
        
        # create horizontal and vertical diameters of rectangle grid
        x = np.linspace(left_point, right_point, int(n))
        y = np.linspace(down_point, up_point, int(n))
        
        # create full rectangle grid of points
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # take points, which inside the circle with radius r
        for i in range(len(positions[0])):
            point = [positions[0][i], positions[1][i]]
            if np.linalg.norm([c, point]) <= r:
                grid[0].append(point[0])
                grid[1].append(point[1])
        
        return grid
    
    def _create_disk_grid_gauss(self, c0, r, n, profile):
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
            Function with given distribution.    
            For example Gauss function.
            
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
        
        return grid

    def _gaus2d(self, x=0, y=0, mx=0, my=0, sx=0.003, sy=0.003, a=1, pad=0):
        return a * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.))) + pad

if __name__=='__main__':
    
    fb = fatbeam(traj_list_optimized[0], E, B, geomT15, Btor, Ipl)
    fb.calc()
    fb.save('fatbeam/results/new')