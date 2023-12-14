# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:11:46 2023

@author: reonid
"""

import itertools
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays


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

class FastGridInterpolator3D:
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy, self.zz = points  
        #self.grid = _mgrid(*points) 
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0],
                             self.zz[1] - self.zz[0]])
        
        self.res3 = self.res[0]*self.res[1]*self.res[2]  
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0], self.zz[ 0] ])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1], self.zz[-1] ])

        self._reginterp = RegularGridInterpolator(points, values, method=method, 
                                                  fill_value=fill_value, bounds_error=False, **kwargs)
         

    def __call__(self,  point): 
#       !!! RegularGridInterpolator is drastically faster in this case
        if hasattr(point[0], '__iter__'):   
            return self._reginterp(point)
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            return self.fill_value # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [self.xx[ii[0]  ], self.yy[ii[1]  ], self.zz[ii[2]  ]  ]
        higher_pt = [self.xx[ii[0]+1], self.yy[ii[1]+1], self.zz[ii[2]+1]  ]

        delta_x = [higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  
        delta_z = [higher_pt[2] - point[2],   point[2] - lower_pt[2] ] 

        number = 0
        weights = [0.]*8
        for i in range(2):
            for j in range (2):
                for k in range(2):
                    weights[number] = delta_x[i]*delta_y[j]*delta_z[k]/self.res3
                    number += 1

#        weights = [0.]*8 
#        for i, j, k in itertools.product(*[[0, 1] for _ in range(3)]):
#            weights[4*i + 2*j + k] = delta_x[i]*delta_y[j]*delta_z[k]/self.res3
            
        #finding interpolation
        v = 0.0
        vv = self.values

        for ijk in range(8):
            i = ( ijk >> 2 ) % 2
            j = ( ijk >> 1 ) % 2
            k =   ijk % 2 
      
            v += weights[ijk]* vv[ii[0] + i, ii[1] + j, ii[2] + k]

#        for i, j, k in itertools.product(*[[0, 1] for _ in range(3)]):
#            v += weights[4*i + 2*j + k]* vv[ii[0] + i, ii[1] + j, ii[2] + k]

#        for ijk, i, j, k in bin_indices3d:
#            v += weights[ijk]* vv[ii[0] + i, ii[1] + j, ii[2] + k]

        return v 

class FastGridInterpolator2D:
    def __init__(self, points, values, method="linear", fill_value=np.nan, **kwargs): 
        self.ndim = len(points)
        self.xx, self.yy = points  
        #self.grid = _mgrid(*points) 
        self.values = values
        self.fill_value = fill_value 
        
        self.res = np.array([self.xx[1] - self.xx[0], 
                             self.yy[1] - self.yy[0]])
        
        self.res2 = self.res[0]*self.res[1]
    
        self.lower_corner =  np.array([ self.xx[ 0], self.yy[ 0]])
        self.upper_corner  = np.array([ self.xx[-1], self.yy[-1]])

        self._reginterp = RegularGridInterpolator(points, values, method=method, 
                                                  fill_value=fill_value, bounds_error=False, **kwargs)
         

    def __call__(self,  point): 
#       !!! RegularGridInterpolator is drastically faster in this case
        if hasattr(point[0], '__iter__'):   
            return self._reginterp(point)
        
        if any(point - self.lower_corner <= 0.0) or any(self.upper_corner - point <= 0.0):  # np.logical_and, np.logical_or
            return self.fill_value # [0]

        ii = (  (point - self.lower_corner)/self.res  ) # // 1
        ii = ii.astype(int) #   ii = np.int_(ii)  #   ii = ii.astype(int, copy=False)
        
        lower_pt  = [  self.xx[ii[0]  ], self.yy[ii[1]  ]  ]
        higher_pt = [  self.xx[ii[0]+1], self.yy[ii[1]+1]  ]

        delta_x = [ higher_pt[0] - point[0],   point[0] - lower_pt[0] ] 
        delta_y = [ higher_pt[1] - point[1],   point[1] - lower_pt[1] ]  

        number = 0
        weights = [0.]*4
        for i in range(2):
            for j in range (2):
                    weights[number] = delta_x[i]*delta_y[j]/self.res2
                    number += 1
            
        #finding interpolation
        v = 0.0
        vv = self.values

        for ij in range(4):
            i = ( ij >> 1 ) % 2
            j =   ij % 2 
      
            v += weights[ij]* vv[ii[0] + i, ii[1] + j]

        return v 


# 2D, 3D ... 
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

        multipoint = xi_T.shape[1] > 1
        
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

        
if __name__ == '__main__': 
    xx = np.linspace(-1.0,   10.0, 100)
    yy = np.linspace(-15.8, 126.0, 120)
    zz = np.linspace(-30.0,  30.0, 130)
    
    x3d, y3d, z3d = np.meshgrid(xx, yy, zz, indexing='ij')  # !!! indexing='ij' - else it swaps x and y
    vv = np.sin(x3d*1.7 + y3d*0.18 + z3d*0.98)

    fastintp = FastGridInterpolator3D  ( (xx, yy, zz), vv, fill_value=0.0)
    regintp  = RegularGridInterpolator ( (xx, yy, zz), vv, fill_value=0.0, bounds_error=False)
    testintp = TestGridInterpolator    ( (xx, yy, zz), vv, fill_value=0.0)
    fastintp2d = FastGridInterpolator2D( (xx, yy),     vv[:,:,0], fill_value=0.0)
    

    pt = np.array([0.2, 0.4, 0.5])
    v   = fastintp(pt)
    _v  = regintp (pt)
    __v = testintp(pt)

    print('fast:    ', v)
    print('reg:   ', _v)
    print('test:  ', __v)
    
    if not np.isclose(v, _v): 
        raise Exception('FastGridInterpolator3D: Test failed')

    if not np.isclose(v, __v): 
        raise Exception('TestGridInterpolator3D: Test failed')


    #2d
    v2d = fastintp2d(pt[0:2])
    _v2d = fastintp2d._reginterp(pt[0:2])
    print('fast2d:  ', v2d)
    print('reg2d: ', _v2d)
    if not np.isclose(v2d, _v2d): 
        raise Exception('TestGridInterpolator2D: Test failed')
    
    
    # grid test
    xx = np.arange(0.0, 5.0, 1.0) 
    yy = np.arange(0.0, 6.0, 1.0) 
    zz = np.arange(0.0, 7.0, 1.0)

    grid = _mgrid(xx, yy, zz)
    _grid = np.mgrid[0.0:5.0:1.0, 
                     0.0:6.0:1.0,
                     0.0:7.0:1.0]
    
    if not np.isclose( np.sum(grid - _grid), 0.0): 
        raise Exception('_mgrid3d: Test failed') 

    grid = _mgrid(xx, yy)
    _grid = np.mgrid[0.0:5.0:1.0, 
                     0.0:6.0:1.0]
    
    if not np.isclose( np.sum(grid - _grid), 0.0): 
        raise Exception('_mgrid2d: Test failed') 

