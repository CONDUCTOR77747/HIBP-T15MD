# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:53:46 2023

@author: reonid
"""

#from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

#import multiprocessing as mp
#from joblib import Parallel, delayed

from .geom import pt3D, vec3D, join_gabarits, calc_gabarits, get_coord_indexes, _regularPolygon3D

def _coils_set(inner, outer, tt, zz): # 0 <= t <= 1
    result = []
    
    for t in tt: 
        for z in zz: 
            coil_array = outer - t*(outer - inner)
            coil_array[:, 2] = z 
            result.append(coil_array)
    
    return result

EPS_WIRE = 0.0005 # ??? m
def SI_calc_B(r, IdLs_of_elem, rr_of_elem, eps=EPS_WIRE):
    rr = r - rr_of_elem
    abs_rr = np.linalg.norm(rr, axis=1)
    
    mask = (abs_rr < eps) # create a mask to exclude points close to wire
    abs_rr3 = abs_rr**3
    rr_rr3 = rr / abs_rr3 [:, np.newaxis] # transpone    rr/(|rr|^3)

    IdLs_x_rr = np.cross(IdLs_of_elem, rr_rr3) 
    IdLs_x_rr[mask] = [0., 0., 0.]

    # claculate sum of contributions from all current elements
    B = np.sum(IdLs_x_rr, axis=0)
    return B *1e-7  # ???

#EPS_WIRE = 0.0005 # m
#def calc_Bpoint(r, IdL, r_elem):
#    r2 = r - r_elem
#    r25 = np.linalg.norm(r2, axis=1)
#    # create a mask to exclude points close to wire
#    mask = (r25 < EPS_WIRE)  # 0.6 is ok for plasma current field
#    r25 = r25**3
#    r3 = r2 / r25 [:, np.newaxis]
#
#    cr = np.cross(IdL, r3)
#    cr[mask] = [0., 0., 0.]
#
#    # claculate sum of contributions from all current elements
#    s = np.sum(cr, axis=0)
#    return s *1e-7


class ThickCoil: 
    def __init__(self, filename, J): 

        self.filename = filename
        self.J = J
        self.elem_radii = None
        self.elem_IdLs = None
        self._gabarits = None

        if filename is not None: # !!! format of coil file can be changed in future !!!

            data = np.loadtxt(filename) / 1000.0  # [m]
            N = data.shape[0]
            zz = np.zeros(N)
    
            # data have only x and y columns, data are closed (first and last points are the same)
            self.outer_coil_array = np.vstack( (data[:, [2, 3]].T, zz) ).T
            self.inner_coil_array = np.vstack( (data[:, [0, 1]].T, zz) ).T
            
            zwidth = 0.196
            dz = zwidth*0.33333
            
            self.filament_arrays = _coils_set(self.inner_coil_array, self.outer_coil_array, 
               #[0.0, 0.33333, 0.66666, 1.0], 
               #[-zwidth*0.5, -zwidth*0.5 + dz, zwidth*0.5 - dz, zwidth*0.5]) 
                  [0.5], 
                  [0.0]) 
    
            self.filamentJ = self.J / len(self.filament_arrays)
        else: 
            # Circular coil
            self.filamentJ = self.J
            circle = _regularPolygon3D(npoints=100, center=pt3D(0, 0, 0), radius = 1.0, normal=vec3D(0, 0, 1), closed=True)
            circle = np.array(circle)
            self.outer_coil_array = circle
            self.inner_coil_array = circle
            self.filament_arrays = [circle]
        
    @classmethod
    def test_cirular_coil(cls, J, center, radius, npoints): 
        result = ThickCoil(None, J)
        circle = _regularPolygon3D(npoints, center, radius, normal=vec3D(0, 0, 1), closed=True)
        circle = np.array(circle)
        result.outer_coil_array = circle
        result.inner_coil_array = circle
        result.filament_arrays = [circle]
        result.filamentJ = J
        return result

        

    def gabarits(self): 
        if self._gabarits is None: 
            for arr in self.filament_arrays: 
                self._gabarits = join_gabarits(self._gabarits, calc_gabarits(arr))
            
        return self._gabarits
        
        

    def transform(self, mx): 
        self.clear_cache()
        for arr in self.filament_arrays:         
            for i in range(arr.shape[0]): 
                arr[i] = mx.dot( arr[i] ) 
                
    def translate(self, vec): 
        self.clear_cache()
        for arr in self.filament_arrays: 
            arr += vec

    def plot(self, axes_code=None, *args, **kwargs):
        X, Y = get_coord_indexes(axes_code)
        for arr in self.filament_arrays: 
            plt.plot(arr[:, X], arr[:, Y], *args,**kwargs)

    def clear_cache(self): 
        self.elem_radii = None
        self.elem_IdLs = None
        self._gabarits = None
        
    #--------------------------------------------------------------------------

    def calcB(self, r): 
        B = vec3D(0, 0, 0)
        for arr in self.filament_arrays: 
            arr0 = arr[0:-1, :]    # !!! correct for closed arrays
            arr1 = arr[1:  , :]
            rr = (arr0 + arr1) * 0.5
            dL = arr1 - arr0
            IdL = dL * self.filamentJ
            B += SI_calc_B(r, IdL, rr)
        return B

    def array_of_IdLs(self): 
        if (self.elem_radii is not None) and (self.elem_IdLs is not None): 
            return self.elem_radii, self.elem_IdLs
        
        self.elem_radii = None
        self.elem_IdLs = None
        
        for arr in self.filament_arrays: 
            arr0 = arr[0:-1, :]    # !!! correct for closed arrays
            arr1 = arr[1:  , :]
            _rr = (arr0 + arr1) * 0.5
            _dL = arr1 - arr0
            _IdL = _dL * self.filamentJ

            if self.elem_radii is None: # and IdL is None
                self.elem_radii = _rr
                self.elem_IdLs = _IdL
            else: 
                self.elem_radii = np.vstack((self.elem_radii, _rr))
                self.elem_IdLs  = np.vstack((self.elem_IdLs, _IdL))
                
        return self.elem_radii, self.elem_IdLs

#

        

