# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:42:34 2023

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt

from hibpcalc.geom import (pt3D, vec3D, size3D, 
                  identMx, invMx, rotateMx, xwardRotateMx, xScaleMx, skewMx, 
                  plot_point, plot_polygon, 
                  calc_gabarits, join_gabarits, inside_gabarits, outside_gabarits, testpoints)

#%%

#class Obj3D: 
#    def __init__(self): 
#        self._gabarits = None

#%%

class Box3D: 
    def __init__(self, xlen, ylen, zlen): 
        # history of transformations
        self._vec = vec3D(0, 0, 0)
        self._mx  = identMx()
        self._imx = identMx()
        # gabarits
        dx, dy, dz = xlen*0.5, ylen*0.5, zlen*0.5
        self._d = size3D(dx, dy, dz)
        self._gabarits = None

        p1 = pt3D( dx,  dy,  dz ) 
        p2 = pt3D(-dx,  dy,  dz )
        p3 = pt3D(-dx, -dy,  dz ) 
        p4 = pt3D( dx, -dy,  dz )
        p5 = pt3D( dx,  dy, -dz ) 
        p6 = pt3D(-dx,  dy, -dz )
        p7 = pt3D(-dx, -dy, -dz ) 
        p8 = pt3D( dx, -dy, -dz )
        
        self._points = [p1, p2, p3, p4, p5, p6, p7, p8]
        self.recalc_gabarits()

    def translate(self, vec): 
        self._vec += vec
        self._points = [pt + vec for pt in self._points]  
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        self._vec = mx.dot(self._vec)
        self._mx  = mx.dot(self._mx)
        self._imx = invMx(self._mx)  
        self._points = [mx.dot(pt) for pt in self._points]
        self.recalc_gabarits()
        return self
    
    def trans(self, *args):
        for a in args: 
            if len(a.shape) == 1: 
                self.translate(a)
            elif len(a.shape) == 2: 
                self.transform(a)
            else: 
                raise Exception('Box3D.trans: invalid argument')
        return self

    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits(self.points() )

    def plot(self, axes_code=None, **kwargs): 
        color = kwargs.get('color')
        if color is None: 
            for pgn in self.polygons(): 
                color = plot_polygon(pgn, axes_code, color=color, **kwargs)[0].get_color()
        else:         
            for pgn in self.polygons(): 
                plot_polygon(pgn, axes_code, **kwargs)
        
    def contains_pt(self, pt): 
        if outside_gabarits(pt, self._gabarits): 
            return False
        
        pt0 = self._imx.dot(pt - self._vec)

#        return all(pt0 >= -self._d) and all( pt0 <= self._d)
    
        # !!! faster
        d = self._d
        return (  (pt0[0] >= -d[0]) and (pt0[1] >= -d[1]) and (pt0[2] >= -d[2]) and 
                  (pt0[0] <=  d[0]) and (pt0[1] <=  d[1]) and (pt0[2] <=  d[2])     )      
            
        
    def points(self): 
        return self._points  # yield from self._points

    def gabarits(self): 
        return self._gabarits

    def tetragon(self, i, j, k, m):
        return [ self._points[i], self._points[j], self._points[k], self._points[m] ]
    
    def polygons(self): 
        yield self.tetragon(0, 3, 7, 4)  # x > 0
        yield self.tetragon(1, 2, 6, 5)  # x < 0 
        yield self.tetragon(0, 1, 5, 4)  # y > 0
        yield self.tetragon(2, 3, 7, 6)  # y < 0 
        yield self.tetragon(0, 1, 2, 3)  # z > 0
        yield self.tetragon(4, 5, 6, 7)  # z < 0 

    def rotate(self, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        self.trans(-pivot_point, mx, pivot_point)
        return self

    def rotate_with_skew(self, fixed_plane, angle): 
        # to emulate Philipp's transformation in plate_flags
        
        # prefix
        pln_pt0, pln_normal = fixed_plane
        xward_mx = xwardRotateMx(pln_normal)
        self.trans(-pln_pt0, xward_mx)
        
        angle_ = np.arctan(np.sin(angle))
        mx = skewMx(angle_, 'YX')
        self.trans(mx)
         
        mx = xScaleMx( np.cos(angle) )
        self.trans(mx)
        
        # postfix
        self.trans(invMx(xward_mx), pln_pt0)
        return self

#%%

class Group3D: 
    def __init__(self, elemlist=[]): 
        self._list = elemlist  # list(elemlist)
        self._gabarits = None
        #self._checkgabarits = True
    
    def append(self, elem): 
        self._list.append(elem)
        self._gabarits = None

    def __iter__(self): 
        return self._list.__iter__()



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



    def contains_pt(self, pt): 
        #if self._checkgabarits: 
        if outside_gabarits(pt, self.gabarits()): 
            return False
        
        for elem in self._list: 
            if elem.contains_pt(pt): 
                return True
        return False
    
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
    
    def elements(self): 
        return self._list 

#    def points(self): 
#        return self._points

#%%

class FastGroup3D(Group3D): # faster gabarits check
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
        if outside_gabarits(pt, self._gabarits): 
            return False

        for elem in self._list: 
            if elem.contains_pt(pt): 
                return True
        return False
    
class BoundlessGroup3D(Group3D): # without gabarits check
    def contains_pt(self, pt): 
        for elem in self._list: 
            if elem.contains_pt(pt): 
                return True
        return False
    
#%% 
        
def box3D(xlim, ylim, zlim): 
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    box = Box3D(xmax-xmin, ymax-ymin, zmax-zmin)
    box.translate( 0.5*vec3D(xmax+xmin, ymax+ymin, zmax+zmin) )
    return box    

def gabaritbox3D(gbr): 
    (xmin, ymin, zmin), (xmax, ymax, zmax) = gbr

    box = Box3D(xmax-xmin, ymax-ymin, zmax-zmin)
    box.translate( 0.5*vec3D(xmax+xmin, ymax+ymin, zmax+zmin) )
    return box    
    

if __name__ == '__main__': 
 
    b = Box3D(3.0, 1.0, 2.0)
    
    v = vec3D(1, 0.5, 0.3)
    mx = rotateMx(vec3D(1.3, 2.4, 0.2), 1.33)

    b.transform(mx)
    b.translate(v)

    plt.figure()
    b.plot('XY', color='r')      #b.plot('XZ')

    gbr = b.gabarits()
    
    testpts = testpoints(gbr, 10000)
    for pt in testpts: 
        if b.contains_pt(pt): 
            plot_point(pt, 'XY', ms=2)
            
    
    
        