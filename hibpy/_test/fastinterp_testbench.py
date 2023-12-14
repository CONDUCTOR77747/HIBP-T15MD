# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:49:00 2023

@author: rounin
"""

import itertools
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

xx = np.linspace(-1.0,   10.0, 100)
yy = np.linspace(-15.8, 126.0, 120)
zz = np.linspace(-30.0,  30.0, 130)
x3d, y3d, z3d = np.meshgrid(xx, yy, zz, indexing='ij')  # !!! indexing='ij' - else it swaps x and y
v3d = np.sin(x3d*1.7 + y3d*0.18 + z3d*0.98)

pt = np.array([0.2, 0.4, 0.5])
pts = pt
for i in range(1000): 
    pts = np.vstack( (pts, pt + 0.01*i) )




#%%
points = (xx, yy, zz)
values = v3d
fill_value = 0.0

xi = ptы

#%% 

xx_yy_etc =  tuple([np.asarray(p) for p in points]) 
ndim = len(xx_yy_etc)
res = np.array([coords[1] - coords[0] for coords in points]) # self.res = np.array([self.xx[1] - self.xx[0], self.yy[1] - self.yy[0], self.zz[1] - self.zz[0]])
#values = values
        
res3 = np.prod(res)                                     
     
lower_corner  = np.array([coords[ 0] for coords in points])  # np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
upper_corner = np.array([coords[-1] for coords in points])  # np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])


#%% __call__(self, xi)
xi = _ndim_coords_from_arrays(xi, ndim=ndim) # tuple -> multidim array
if xi.shape[-1] != len(xx_yy_etc):
    raise ValueError("Dimension error %d != %d" % (xi.shape[1], ndim))

xi_shape = xi.shape
xi = xi.reshape(-1, xi_shape[-1])


#%% indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
xi_T = xi.T

# find relevant edges between which xi are situated
multipoint = xi_T.shape[1] > 1

indices = []
# compute distance to lower edge in unity units
norm_distances = []
# check for out of bounds xi
out_of_bounds = np.zeros((xi_T.shape[1]), dtype=bool)

print(multipoint)
# iterate through dimensions
for k, (x, xgrid) in enumerate(zip(xi_T, xx_yy_etc)):
    
    #i = np.searchsorted(xgrid, x) - 1
    if multipoint:   
       i = np.searchsorted(xgrid, x) - 1
    else:
       i = (   (x - lower_corner[k])/res[k]   ).astype(int)
    
    i[i < 0] = 0
    i[i > xgrid.size - 2] = xgrid.size - 2
    indices.append(i)
    norm_distances.append( (x - xgrid[i]) / (xgrid[i + 1] - xgrid[i]) )

    out_of_bounds += x < xgrid[0]
    out_of_bounds += x > xgrid[-1]
        
#return indices, norm_distances, out_of_bounds

#%% result = _evaluate_linear(indices, norm_distances)

# slice for broadcasting over trailing dimensions in self.values
vslice = (slice(None),) + (None,)*(values.ndim - len(indices))

# find relevant values
# each i and i+1 represents a edge

edges = itertools.product(*[[i, i + 1] for i in indices])
_values = 0.0
for edge_indices in edges:
    weight = 1.0
    for ei, i, normi in zip(edge_indices, indices, norm_distances):
        weight *= np.where(ei == i, 1.0 - normi, normi) 

    _values += np.asarray(values[edge_indices]) * weight[vslice]

#return values
result = _values

#%%

result[out_of_bounds] = fill_value

result = result.reshape(xi_shape[:-1] + values.shape[ndim:])

print(result)



#%% 

class TestGridInterpolator: # modified code from RegularGridInterpolator

    def __init__(self, points, values, fill_value=np.nan, *kwargs):
        self.ndim = len(points)
        self.fill_value = fill_value
        self.xx_yy_etc = tuple([np.asarray(p) for p in points])  # self.grid = ...
  
        self.values = values
        
        #self.xx, self.yy, self.zz = points
        self.res = np.array([coords[1] - coords[0] for coords in points]) # self.res = np.array([self.xx[1] - self.xx[0], self.yy[1] - self.yy[0], self.zz[1] - self.zz[0]])
        self.res3 = np.prod(self.res)                                     # self.res[0]*self.res[1]*self.res[2]  
     
        self.lower_corner  = np.array([coords[ 0] for coords in points])  # np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
        self.upper_corner_ = np.array([coords[-1] for coords in points])  # np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])
        self.upper_corner = self.upper_corner_ + self.res

    def __call__(self, xi):
        xi = _ndim_coords_from_arrays(xi, ndim=self.ndim) # tuple -> multidim array
        if xi.shape[-1] != len(self.xx_yy_etc):
            raise ValueError("Dimension error %d != %d" % (xi.shape[1], self.ndim))
        
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)

        #indices, norm_distances, out_of_bounds = self.__find_indices(xi)             

        result = self._evaluate_linear(indices, norm_distances)
        result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[self.ndim:])

    def __call__new(self, xi):
        indices, norm_distances = self.__find_indices(xi)             

        result = self.__evaluate_linear(indices, norm_distances)
        #result[out_of_bounds] = self.fill_value
         
        if isinstance(result, np.ndarray): 
            return result.reshape(xi_shape[:-1] + self.values.shape[self.ndim:])
        else: 
            return result

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for ei, i, normi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1.0 - normi, normi) 
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def __evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents an edge
        edges = itertools.product(*[[i, i + 1] for i in indices])

        values = 0.0
        for edge_indices in edges:
            weight = 1.0
            for edgei, i, normi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(edgei == i, 1.0 - normi, normi) 
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values

    def _find_indices(self, xi_T):   # indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        # find relevant edges between which xi are situated

        multipoint = hasattr(xi_T[0], '__iter__')
        
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi_T.shape[1]), dtype=bool)

        # iterate through dimensions
        for k, (x, xgrid) in enumerate(zip(xi_T, self.xx_yy_etc)):
            
            if multipoint:   
               i = np.searchsorted(xgrid, x) - 1
            else:
               i = (   (x - self.lower_corner[k])/self.res[k]   ).astype(int)
            
            i[i < 0] = 0
            i[i > xgrid.size - 2] = xgrid.size - 2
            indices.append(i)
            norm_distances.append( (x - xgrid[i]) / (xgrid[i + 1] - xgrid[i]) )

            out_of_bounds += x < xgrid[0]
            out_of_bounds += x > xgrid[-1]
                
        return indices, norm_distances, out_of_bounds
    
    def __find_indices(self, xi):   
        # find relevant edges between which xi are situated

        _indices = (  (xi - self.lower_corner)/self.res  ) # // 1
        norm_distances = _indices % 1
        indices = _indices.astype(int) 
        
        return indices, norm_distances
