# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:45:44 2023

@author: reonid
"""

import numpy as np
#import matplotlib.pyplot as plt

from .geom import vec3D, outside_gabarits, join_gabarits


#%%

class Group3D: 
    def __init__(self, elemlist=None): 
        if elemlist is None:
            elemlist = []

        self._list = elemlist  # list(elemlist)
        self._gabarits = None
        self._active_elem = None
        #self._checkgabarits = True
         
    def append(self, elem): 
        self._list.append(elem)
        self._gabarits = None

    def remove(self, elem): 
        self._list.remove(elem)
        self._gabarits = None

    def __iter__(self): 
        return self._list.__iter__()

    def __getitem__(self, arg): 
        return self._list[arg]

    def elements(self): 
        return self._list 
        

    def translate(self, vec): 
        for elem in self._list: 
            elem.translate(vec)
        self._gabarits = None
        return self

    def transform(self, mx): 
        for elem in self._list: 
            elem.transform(mx)
        self._gabarits = None
        return self
    
    def trans(self, *args):
        for elem in self._list: 
            elem.trans(*args)
        self._gabarits = None
        return self

    def rotate(self, pivot_point, axis, angle): 
        for elem in self._list: 
            elem.rotate(pivot_point, axis, angle)
        self._gabarits = None
        return self


    def plot(self, axes_code=None, **kwargs): 
        for elem in self._list: 
            elem.plot(axes_code, **kwargs)
        return self


    def recalc_gabarits(self): 
        self._gabarits = None # calc_gabarits(self.points() )   
        for elem in self._list: 
            self._gabarits = join_gabarits(self._gabarits, elem.gabarits())

    def gabarits(self): 
        if self._gabarits is None: 
            self.recalc_gabarits()
        return self._gabarits

    #--------------------------------------------------------------------------

    # electrostatic geometry: for initial conditions of Laplace eq.  
    def contains_pt(self, pt):    
        #if self._checkgabarits: 
        self._active_elem = None
        
        if outside_gabarits(pt, self.gabarits()): 
            return False
        
        for elem in self._list: 
            if elem.contains_pt(pt): 
                self._active_elem = elem
                return True
        return False

    # obstacle geometry: trajectory passing
    def intersect_with_segment(self, pt0, pt1): 
        self._active_elem = None
        
        for elem in self._list: 
            intersect_pt = elem.intersect_with_segment(pt0, pt1)
            if intersect_pt is not None: 
                self._active_elem = elem
                return intersect_pt
        return None

    # magnetic geometry
    def calcB(self, r): 
        B = vec3D(0, 0, 0)
        for elem in self._list: 
            B += elem.calcB(r)
            
        return B    

    def array_of_IdLs(self): 
        rr = None
        IdL = None
        for elem in self._list: 
            _rr, _IdL = elem.array_of_IdLs() # 
            
            rr  = np.vstack((rr,  _rr )) if rr  is not None else _rr
            IdL = np.vstack((IdL, _IdL)) if IdL is not None else _IdL
        
        return rr, IdL

#    def points(self): 
#        return self._points

#%%

class FastGroup3D(Group3D): # faster gabarits check
    
    def append(self, elem): 
        self._list.append(elem)
        self.recalc_gabarits()

    def remove(self, elem): 
        self._list.remove(elem)
        self.recalc_gabarits()



    def translate(self, vec): 
        for elem in self._list: 
            elem.translate(vec)
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        for elem in self._list: 
            elem.transform(mx)
        self.recalc_gabarits()
        return self
    
    def trans(self, *args):
        for elem in self._list: 
            elem.trans(*args)
        self.recalc_gabarits()
        return self

    def rotate(self, pivot_point, axis, angle): 
        for elem in self._list: 
            elem.rotate(pivot_point, axis, angle)
        self.recalc_gabarits()
        return self    




    def contains_pt(self, pt): 
        self._active_elem = None
        
        if outside_gabarits(pt, self._gabarits): #  !!! self._gabarits always should be correct !!!
            return False

        for elem in self._list: 
            if elem.contains_pt(pt): 
                self._active_elem = elem
                return True
        return False

#%%    

class BoundlessGroup3D(Group3D):

    def contains_pt(self, pt):   
        # without gabarits check
        self._active_elem = None
        
        for elem in self._list: 
            if elem.contains_pt(pt): 
                self._active_elem = elem
                return True
        return False

