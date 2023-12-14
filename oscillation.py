# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:21:32 2023

@author: Krohalev_OD
"""

#%% imports
import os
import copy
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from hibpcalc.fields import array_to_value
from hibpcalc.geomfunc import rotate, ray_segm_intersect_2d

#%% additional functions
def parabolic(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return (1 - 4*rho_loc**2/width**2)

def bell(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return (1 - 4*(abs(rho_loc)/width)**2.5)**5.5

def bell_wide(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return 1/(1 + 4*(abs(rho_loc)/width)**2)**(4./3.)

def bell_peaked(rho_loc, width=2.0):
    if abs(rho_loc) > width/2:
        return 0.
    else:
        return 1/(1 + 4*(0.8*abs(rho_loc)/width)**2)**6.0
    
def f_with_base(f, v_edge):
    def g(rho, width=2.0):        
        if abs(rho) > width/2:
            return 0.
        else:
            v0 = f(0.0, width)
            y = f(rho, width)
            y = v0 -  (v0 - y)*(v0 - v_edge)
            return y
    return np.vectorize(g)

#%% coordinates converter
class CoordConverterDShapeSimple():
    def __init__(self, R, separatrix):
        self.R = R
        self.sep = separatrix
        pass
    
    def __call__(self, r):
        phi = self.calc_phi(r)
        rho, theta = self.calc_rho_and_theta(r, phi)
        return rho, theta, phi
    
    def calc_R_vector(self, r):
        x, y, z = r
        R_v = np.array([x, 0, z])/np.sqrt(x**2+z**2)*self.R
        return R_v
    
    def calc_phi(self, r):
        R_v = self.calc_R_vector(r)
        phi = np.arcsin(R_v[2]/self.R)
        # if R_v[2] < 0:
        #     phi = phi - np.pi
        return phi
    
    def calc_rho_and_theta(self, r, phi):
        r = np.array(r)
        r_eq = rotate(r, deg=-np.degrees(phi))
        r_eq[2] = 0. # it should be 0. after rotation, but to avoid computational mistakes we set it 0. additionally
        r_eq = r_eq[0:2]
        r_center = [self.R, 0]
        r_local = r_eq - r_center
        if np.all(np.isclose(r_local, [0., 0.])):
            return 0., 0.
        theta = math.atan2(r_local[1], r_local[0])
        for segm_0, segm_1 in zip(self.sep[:-1], self.sep[1:]):
            rho = ray_segm_intersect_2d((r_center, r_eq), (segm_0, segm_1))
            if rho is not None:
                break
        else:
            return 0., 0.
        return rho, theta

#%% 
def load_coord(grid, converter, name=None): 
    
    if name is None:
        name = 'coords_on_grid.pkl'
        
    if os.path.exists(name): 
        with open(name, 'rb') as file:
            coords_on_grid = pickle.load(file)
            coords = coords_on_grid['coords']
            loaded_grid = coords_on_grid['grid']
                    
        if np.all(loaded_grid == grid): 
            return coords
        else:
            os.remove(name)
    
    coords = np.zeros_like(grid)
    for i, x in enumerate(grid[0, :, 0, 0]):
        for j, y in enumerate(grid[1, 0, :, 0]):
            for k, z in enumerate(grid[2, 0, 0, :]):
                rho, theta, phi = converter([x, y, z]) 
                coords[:, i, j, k] = rho, theta, phi
    
    coords_on_grid = {'coords': coords, 'grid': grid}
    with open(name, 'wb') as file:
        pickle.dump(coords_on_grid, file)
    return coords

class ValueOnGrid():
    def __init__(self, grid, value_profile, converter, default=np.nan):
        self.grid = grid
        self.res = self.grid[:, 1, 1, 1] - self.grid[:, 0, 0, 0]
        self.res = array_to_value(self.res)
        self.default_value = default
        self.values = np.zeros_like(grid[0])
        self.converter = converter
        self.rhos, self.thetas, self.phis = load_coord(self.grid, self.converter)
        self.value_profile = value_profile
        self.value_profile.add_to_grid(self.grid, self.values, self.rhos, 
                                  self.thetas, self.phis, self.converter)
        self.volume_corner1 = self.grid[:, 0, 0, 0]
        self.volume_corner2 = self.grid[:, -1, -1, -1]
        
    def __call__(self, point):
        '''

        Parameters
        ----------
        point : array of floats, 3 values
            radius-vector of given point

        Returns
        -------
        dv : float
            relative deviation of value in given point

        '''
        #if point is outside the volume - return np.nan
        if any(point - self.volume_corner1 <= 0.0) or any(self.volume_corner2 - point <= 0.0):
            return self.default_value
        
        #finding indexes of left corner of volume with point
        indexes_float = (point - self.volume_corner1)/self.res // 1
        indexes = [[0]*3]*8
        for i in range(3):
            indexes[0][i] = int(indexes_float[i])
        
        # finding weights for all dots close to point
        '''
        delta_x = [x2 - x, x - x1]
        point = [x, y, z]
        '''
        
        i00 = indexes[0][0]
        j01 = indexes[0][1]
        k02 = indexes[0][2]
        
        left_bottom = self.grid[:, i00,     j01,     k02]        
        right_top   = self.grid[:, i00 + 1, j01 + 1, k02 + 1]
        #delta = (right_top - point,   point - left_bottom)
        
        delta_x = [right_top[0] - point[0],   point[0] - left_bottom[0] ] 
        delta_y = [right_top[1] - point[1],   point[1] - left_bottom[1] ]  
        delta_z = [right_top[2] - point[2],   point[2] - left_bottom[2] ] 
        
        res_cubic = self.res**3
        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/res_cubic
                    number += 1
        
        #finding interpolation
        dv = 0.0
        
        _dv = self.values

        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
            
            dv += weights[ijk]* _dv[i00 + i, j01 + j, k02 + k]

        return dv
    
    def __add__(self, other):
        if not isinstance(other, ValueOnGrid):
            raise ValueError(f'added objects must both be ValueOnGrid or its subclasses, {type(other)} is given')
        
        new_value = copy.deepcopy(self)
        
        same_grid = False
        try:
            same_grid = np.all(new_value.grid == other.grid)
        except:
            pass
        
        if same_grid:
            new_value.values += other.values
        else:
            for i, x in enumerate(new_value.grid[0, :, 0, 0]):
                for j, y in enumerate(new_value.grid[1, 0, :, 0]):
                    for k, z in enumerate(new_value.grid[2, 0, 0, :]):
                        r = [x, y, z]
                        new_value.values[i, j, k] += other.values(r)
                        
        return new_value
    
    def __sub__(self, other):
        if not isinstance(other, ValueOnGrid):
            raise ValueError(f'added objects must both be ValueOnGrid or its subclasses, {type(other)} is given')
        
        new_value = copy.deepcopy(self)
        
        same_grid = False
        try:
            same_grid = np.all(new_value.grid == other.grid)
        except:
            pass
        
        if same_grid:
            new_value.values -= other.values
        else:
            for i, x in enumerate(new_value.grid[0, :, 0, 0]):
                for j, y in enumerate(new_value.grid[1, 0, :, 0]):
                    for k, z in enumerate(new_value.grid[2, 0, 0, :]):
                        r = [x, y, z]
                        new_value.values[i, j, k] -= other.values(r)
                        
        return new_value
    
    def __mul__(self, other):
        if not isinstance(other, ValueOnGrid):
            raise ValueError(f'added objects must both be ValueOnGrid or its subclasses, {type(other)} is given')
        
        new_value = copy.deepcopy(self)
        
        same_grid = False
        try:
            same_grid = np.all(new_value.grid == other.grid)
        except:
            pass
        
        if same_grid:
            new_value.values = new_value.values*other.values
        else:
            for i, x in enumerate(new_value.grid[0, :, 0, 0]):
                for j, y in enumerate(new_value.grid[1, 0, :, 0]):
                    for k, z in enumerate(new_value.grid[2, 0, 0, :]):
                        r = [x, y, z]
                        new_value.values[i, j, k] = new_value.values[i, j, k]*other.values(r)
                        
        return new_value
    
    def plot(self, xx, yy, ax=None, axes='XY'):
        z = 0.
        values = np.zeros((len(xx), len(yy)))
        for i, x in enumerate(xx):
            for j, y in enumerate(yy):
                value = self([x, y, z])
                values[i, j] = value
        plt.figure()
        plt.imshow(values.T, extent=[xx[0], xx[-1], yy[0], yy[-1]])
        

#%% oscillation
class OscillationProfile():
    def __init__(self, m, n, position, width, A, omega=0.0, f=parabolic):
        '''
        
        Parameters
        ----------
        
        m : int
            poloidal mode number
        n : int
            toroidal mode number
        position : float
            [] rho positions of osc. semi-circle inner border
        width : float
            [] width of osc. circle along minor radius in rho units
        A : float
            [%] magnitude of osc., % of local value
        
        Returns
        -------
        None.
        
        '''
        self.m = m
        self.n = n
        self.pos = position
        self.width = width
        self.mag = A/100
        self.omega = omega
        self.f = f
        
    def __call__(self, rho, theta, phi, t=0.0):
        rho_loc = abs(rho - (self.pos + self.width/2))
        return self.mag*np.cos(self.m*theta + self.n*phi - self.omega*t)*self.f(rho_loc, self.width)

    def add_to_grid(self, grid, dest, rhos, thetas, phis, converter): 
        for i, x in enumerate(grid[0, :, 0, 0]):
            for j, y in enumerate(grid[1, 0, :, 0]):
                for k, z in enumerate(grid[2, 0, 0, :]):
                    rho, theta, phi = rhos[i, j, k], thetas[i, j, k], phis[i, j, k] # converter(np.array([x, y, z]))
                    dest[i, j, k] += self(rho, theta, phi)
                    rhos[i, j, k] = rho
                    thetas[i, j, k] = theta
                    phis[i, j, k] = phi

class OscillationOnGrid(ValueOnGrid):
    def rotate(self, new_phase_pol, new_phase_tor=0.):
        if not hasattr(self, 'phase_pol'):
            self.phase_pol = 0
            self.phase_tor = 0
        for i, x in enumerate(self.grid[0, :, 0, 0]):
            for j, y in enumerate(self.grid[1, 0, :, 0]):
                for k, z in enumerate(self.grid[2, 0, 0, :]):
                    r = [x, y, z]
                    rho, theta, phi = self.converter(r)
                    rot_pol = np.cos(self.m(theta + new_phase_pol))/np.cos(self.m(theta + self.phase_pol)) #!!!
                    rot_tor = np.cos(self.n(phi   + new_phase_tor))/np.cos(self.n(phi   + self.phase_tor)) #!!!
                    self.values[i, j, k] = self.values[i, j, k]*rot_pol*rot_tor #!!!
        
        self.phase_pol = new_phase_pol
        self.phase_tor = new_phase_tor
    
#%% profile
class Profile():
    
    def __init__(self, A0=1, f=parabolic):
        self.f = f
        self.A0 = A0 #значение в центре
    
    #def __call__(self, rho, theta, phi):
    def __call__(self, rho, *args):    
        if rho is None:
            return 0.
        return self.f(rho)*self.A0
    
    def at_rho(self, rho):
        f = np.vectorize(self.f)
        return f(rho)*self.A0
    
    def add_to_grid(self, grid, dest, rhos, thetas, phis, converter):
        for i, x in enumerate(grid[0, :, 0, 0]):
            for j, y in enumerate(grid[1, 0, :, 0]):
                for k, z in enumerate(grid[2, 0, 0, :]):
                    rho, theta, phi = rhos[i, j, k], thetas[i, j, k], phis[i, j, k]  # converter(np.array([x, y, z]))
                    dest[i, j, k] += self(rho, theta, phi)
                    rhos[i, j, k] = rho
                    thetas[i, j, k] = theta
                    phis[i, j, k] = phi
    
class ProfileOnGrid(ValueOnGrid):
    
    def modulate(self, oscillation):
        '''
        Adds deviation of value, caused by given oscillation. All changes 
        applieds to deepcopy of self.
        
        Parameters
        ----------
        oscillation : OscillationOnGrid
            oscillation of chosen value

        Returns
        -------
        new_profile : Profile
            deepcopy of self with applied modulation

        '''
   
        return self + self*oscillation

        
#%% geometry