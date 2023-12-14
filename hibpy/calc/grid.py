# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:13:31 2023

@author: reonid

'G-form'
Standard grid shape (3, Nx, Ny, Nz) - due to compatibility with 
  np.mgrid

'B-form'
Field3D has another shape (Nx, Ny, Nz, 3) - due to compatibility with 
  Philipp's magfieldPF*.npy files

Actually 'G-form' is more convenient for fields: in case of 'G-form'
  Bx, By, Bz = B[0], B[1], B[2]

Raw representation for both cases is array of points with shape (Nx*Ny*Ny, 3)

"""

import numpy as np
#import matplotlib.pyplot as plt

try:
    import multiprocessing as mp
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False


def _mgrid(*xx_yy_etc): # _mgrid(xx, yy), _mgrid(xx, yy, zz)
    ndim = len(xx_yy_etc)
    xNd_yNd_etc = np.meshgrid(*xx_yy_etc, indexing='ij') # x2d_y2d, x2d_y3d_z3d
    
    gridshape = [ndim] + list( xNd_yNd_etc[0].shape )   #gridshape.insert(0, ndim)
    grid = np.empty( tuple(gridshape) )

    for i, coord_array in enumerate(xNd_yNd_etc): 
        #grid[i, :, :, :] = coord_array
        #grid[i, :, :] = coord_array
        grid[i, ...] = coord_array
        
    return grid


#%%    

def _grid_to_raw_points(grid):  # "G-form" grid with shape (3, Nx, Ny, Nz)
    #return np.vstack(map(np.ravel, grid)).T  # drastically slower
    pt_dim = grid.shape[0]
    N = np.prod(grid.shape[1:])
    return grid.reshape(pt_dim, N).swapaxes(0, 1)  # -> (Nx*Ny*Nz, 3) 

def _grid_from_raw_points(points, main_shape):
    pt_dim = points.shape[-1]
    return points.swapaxes(0, 1).reshape( (pt_dim,) + main_shape ) # -> (3, Nx, Ny, Nz) 


#%% Vector field

def _field_to_raw_repr(field): # "B-form" field with shape (Nx, Ny, Nz, 3)
    pt_dim = field.shape[-1]
    N = np.prod(field.shape[0:-1])
    return field.reshape(N, pt_dim) # -> (Nx*Ny*Nz, 3) 
        

def _field_from_raw_repr(raw_field, main_shape):  
    pt_dim = raw_field.shape[-1]     
    return raw_field.reshape(main_shape + (pt_dim,)) # -> (Nx, Ny, Nz, 3) 


def _field_to_G_repr(field): # "B-form" field with shape (Nx, Ny, Nz, 3)
    main_shape = field.shape[0:-1]
    raw_field = _field_to_raw_repr(field)    
    return _grid_from_raw_points(raw_field, main_shape) # -> (3, Nx, Ny, Nz) 


#%%
    
class Grid: 
    def __init__(self, mgrid): 
        self.grid = mgrid

    def __getitem__(self, obj): 
        return self.grid[obj]
    
    @property 
    def shape(self): 
        return self.grid.shape
    
    def as_raw_points(self): 
        return _grid_to_raw_points(self.grid)
    
    @classmethod
    def from_indexing(cls, obj): 
        _grid = np.mgrid[obj]
        return cls(_grid)
 
    @classmethod
    def from_domain(cls, lower_corner, upper_corner, resolution): 
        # create grid of points
        upper_corner += resolution
        _grid = np.mgrid[lower_corner[0]:upper_corner[0]:resolution,
                         lower_corner[1]:upper_corner[1]:resolution,
                         lower_corner[2]:upper_corner[2]:resolution]
            
        return cls(_grid)
    
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            lower_corner = np.array([float(i) for i in f.readline().split()[0:3]])
            upper_corner = np.array([float(i) for i in f.readline().split()[0:3]])
            resolution = float(f.readline().split()[0])
        return cls.from_domain(lower_corner, upper_corner, resolution)
    
    #def from_raw_points(cls, points)

    def calc_raw_field(self, func, parallel=False): 
        points = self.as_raw_points()
        
        if parallel and JOBLIB_AVAILABLE: 
            n_workers = mp.cpu_count() - 1
            raw_field = Parallel (n_jobs=n_workers) (delayed(func)(r) for r in points)            
        else: 
            raw_field = [func(r) for r in points]

        return np.array(raw_field)

    def calc_vector_field(self, func, parallel=False): 
        raw_field = self.calc_raw_field(func, parallel)
        return _field_from_raw_repr(raw_field, self.grid.shape[1:])

    def calc_scalar_field(self, func, parallel=False): 
        raw_field = self.calc_raw_field(func, parallel)
        return raw_field.reshape( self.grid.shape[1:] )

