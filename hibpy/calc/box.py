# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:42:34 2023

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt

from .geom import (pt3D, vec3D, size3D, 
                  identMx, invMx, rotateMx, xwardRotateMx, xScaleMx, skewMx, 
                  plot_point, plot_polygon, 
                  _ptInPolygon3D_, _intersect_plane_segm_, 
                  calc_gabarits, join_gabarits, outside_gabarits, testpoints)    # inside_gabarits

#%%

#class Obj3D: 
#    def __init__(self): 
#        self._gabarits = None

#%%

def _trans_(obj3D, *args): 
        for a in args: 
            if len(a.shape) == 1: 
                obj3D.translate(a)
            elif len(a.shape) == 2: 
                obj3D.transform(a)
            else: 
                raise Exception('_trans_: invalid argument')
        return obj3D

def _rotate_(obj3D, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        _trans_(obj3D, -pivot_point, mx, pivot_point)
        return obj3D


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
        return _trans_(self, *args)

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

#        return np.all(pt0 >= -self._d) and np.all( pt0 <= self._d)
    
        # !!! faster
        d = self._d
        return (  (pt0[0] >= -d[0]) and (pt0[1] >= -d[1]) and (pt0[2] >= -d[2]) and 
                  (pt0[0] <=  d[0]) and (pt0[1] <=  d[1]) and (pt0[2] <=  d[2])     )      
            

    def intersect_with_segment(self, pt0, pt1): 
        if self.contains_pt(pt0) ^ self.contains_pt(pt1):  # ??? > >= < <=  if both points are the same? or on the boundary plane?
            for pgn in self.polygons: 
                r_intersect = pgn.intersect_with_segment(pt0, pt1)
                if r_intersect is not None: 
                    return r_intersect
        return None
        
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

class Polygon3D: 
    def __init__(self, points):
        self._points = points
        self._center = 0.25*(points[0] + points[1] + points[2] + points[3]) # ??? temp
        self._normal = self.calc_normal()
        self._gabarits = None
        self.recalc_gabarits()
    
    def calc_normal(self): 
        n3 = len(self._points) // 3
        p1, p2, p3 = self._points[0], self._points[n3], self._points[n3*2]
        return np.cross(p2 - p1, p3 - p1)
 
    def points(self): 
        return self._points  # yield from self._points       

    def intersect_with_segment(self, pt0, pt1): 
        intersect_pt, t = _intersect_plane_segm_((self._center, self._normal), (pt0, pt1))
        if intersect_pt is None: 
            return None
        elif (t < 0.0)or(t > 1.0): 
            return None        
        elif _ptInPolygon3D_(self._points, intersect_pt): 
            return intersect_pt
        else: 
            return None

    def translate(self, vec): 
        self._points = [pt + vec for pt in self._points]
        self._center += vec
        #self._normal = self._normal
        self.recalc_gabarits()
        return self

    def transform(self, mx): 
        self._points = [mx.dot(pt) for pt in self._points]
        self._center = mx.dot(self._center)
        self._normal = self.calc_normal()  # mx.dot(self._normal) # ??? only for rotation !!! not for skew
        self.recalc_gabarits()
        return self

    def trans(self, *args):
        return _trans_(self, *args)

    def rotate(self, pivot_point, axis, angle): 
        mx = rotateMx(axis, angle)
        self.trans(-pivot_point, mx, pivot_point)
        return self
    
    def recalc_gabarits(self): 
        self._gabarits = calc_gabarits(self.points() )

    def plot(self, **kwargs): 
        plot_polygon(self._points, **kwargs)


class StopPlane: 
    def init(self, point, normal): 
        self._point = point
        self.__normal = normal
    
    def intersect_with_segment(self, pt0, pt1): 
        intersect_pt, t = _intersect_plane_segm_((self._center, self._normal), (pt0, pt1))
        if 0.0 <= t <= 1.0: 
            return intersect_pt
        else: 
            return None


#%% 
        
def limitbox3D(xlim, ylim, zlim): 
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

    
    plt.figure()
    pgns = list( b.polygons() )
    
    pgn = Polygon3D(pgns[0])
    pgn.plot()
    