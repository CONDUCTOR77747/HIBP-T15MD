# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:54:51 2023

@author: reonid
"""

#%% from reometry

import numpy as np
import matplotlib.pyplot as plt

def vec3D(x, y, z):
    return np.array([x, y, z], dtype=np.float64)

pt3D = vec3D
size3D = vec3D

def invMx(mx): 
    return np.matrix(mx).I.A


def identMx():
    return np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0,  1.0,  0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)
    
def scaleMx(k):
    return np.array([[ k,    0.0,  0.0],
                     [ 0.0,  k,    0.0],
                     [ 0.0,  0.0,  k  ]],   dtype=np.float64)

def xScaleMx(k):
    return np.array([[ k,    0.0,  0.0],
                     [ 0.0,  1.0,  0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)


def xySkewMx(ang): #  skewMx(ang, 'XY')
    tg = np.tan(ang)
    return np.array([[ 1.0,  tg,   0.0],    # [0, 1]
                     [ 0.0,  1.0,  0.0],    
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)    

def yxSkewMx(ang): #  skewMx(ang, 'YX')
    tg = np.tan(ang)
    return np.array([[ 1.0,  0.0,  0.0],
                     [ tg,   1.0,  0.0],   # [1, 0]
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)    

def skewMx(ang, plane_code): # 'XY', 'YX', ...
    X, Y = get_coord_indexes(plane_code)
    tg = np.tan(ang)
    mx = identMx()
    mx[X, Y] = tg
    return mx
    
    
def rotateMx(axis, ang):    
    res = identMx()
    s = np.sin(ang)
    c = np.cos(ang)
    x, y, z = -axis/np.linalg.norm(axis)   
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    res[0, 0] = xx + (1 - xx)*c
    res[0, 1] = xy*(1 - c) + z*s
    res[0, 2] = xz*(1 - c) - y*s

    res[1, 0] = xy*(1 - c) - z*s
    res[1, 1] = yy + (1 - yy)*c
    res[1, 2] = yz*(1 - c) + x*s

    res[2, 0] = xz*(1 - c) + y*s
    res[2, 1] = yz*(1 - c) - x*s
    res[2, 2] = zz + (1 - zz)*c

    return res

def xRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)
    return np.array([[ 1.0,  0.0,  0.0],
                     [ 0.0,    c,   -s],
                     [ 0.0,    s,    c]],   dtype=np.float64)

def yRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)    
    return np.array([[ c,    0.0,    s],
                     [ 0.0,  1.0,  0.0],
                     [-s,    0.0,    c]],   dtype=np.float64)

def zRotateMx(ang):
    s = np.sin(ang)
    c = np.cos(ang)
    return np.array([[ c,   -s,    0.0],
                     [ s,    c,    0.0],
                     [ 0.0,  0.0,  1.0]],   dtype=np.float64)


    
def transformPt(pt, mx):   # 2D, 3D
    return mx.dot(pt)

def zwardRotateMx(vec):    # vec -> (0, 0, L)
# to transform any plane to XY plane
    v = vec
    a1 = np.arctan2(v[0], v[2])
    mx1 = yRotateMx(-a1)

    v = transformPt(v, mx1)
    a2 = np.arctan2(v[1], v[2])
    mx2 = xRotateMx(a2)  

    return mx2.dot(mx1)

def xwardRotateMx(vec):    # vec -> (L, 0, 0)
# to transform any plane to XY plane
    v = vec
    a1 = np.arctan2(v[1], v[0])
    mx1 = zRotateMx(-a1)

    v = transformPt(v, mx1)
    a2 = np.arctan2(v[2], v[0])
    mx2 = yRotateMx(a2)  

    return mx2.dot(mx1)

vNorm = np.linalg.norm

def normalized_vector(v): 
    return v/vNorm(v) 

def maxY_IdxAndVal(P, i, j, Y):
    if P[i][Y] > P[j][Y] :   return (i, P[i][Y])
    else:                    return (j, P[j][Y])

def minY_onlyVal(P, i, j, Y):
    if P[i][Y] < P[j][Y] :   return P[i][Y]
    else:                    return P[j][Y]

def ptInPolygon2D(P, pt, X=0, Y=1): 
    '''
    P - point list

    '''
    count = len(P)
    intersect = 0
    #X = 0
    #Y = 1    
    for i in range(count): 
      j = (i+1) % count
      if ( not(    ( pt[Y] < P[i][Y] )and( pt[Y] < P[j][Y] )   )and
           not(    ( pt[Y] > P[i][Y] )and( pt[Y] > P[j][Y] )   )and
           not(      P[i][Y] ==  P[j][Y]                       )    ):
        (k, maxY) = maxY_IdxAndVal(P, i, j)
        
        if maxY == pt[Y]: 
            if P[k][X] > pt[X]: 
               intersect += 1 
        else:  
          # if (not ( min(P[i][Y], P[j][Y]) == pt[Y] )): # ??? why not? 
            if (not ( minY_onlyVal(P, i, j) == pt[Y] )): 
               t = (pt[Y] - P[i][Y] )/( P[j][Y] - P[i][Y] )
               if (  (t > 0.0)and
                     (t < 1.0)and
                     (P[i][X] + t*(P[j][X]-P[i][X]) > pt[X])  ): 
                   intersect +=1
    
    return intersect % 2 == 1 


def _ptInPolygon3D_(P, pt): #  ???
    '''no check if point is in the plane of polygon 
    This function is purposed only for the points that 
    lie on the plane of the polygon
    '''
    normal = np.cross( P[1] - P[0], P[-1] - P[0])
    k = np.argmax(normal)
    X, Y = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[k]
    return ptInPolygon2D(P, pt, X, Y)

#%% 

def calc_gabarits(points): 
    xx = [pt[0] for pt in points]
    yy = [pt[1] for pt in points]
    zz = [pt[2] for pt in points]
    
    xmin, xmax = np.min(xx), np.max(xx)
    ymin, ymax = np.min(yy), np.max(yy)
    zmin, zmax = np.min(zz), np.max(zz)
    
    return np.array([[xmin, ymin, zmin], 
                     [xmax, ymax, zmax]])

def join_gabarits(gbr1, gbr2): 
    if gbr1 is None: 
        return gbr2
    elif gbr2 is None: 
        return gbr1
    
    mins = np.min(   np.vstack((gbr1[0, :], gbr2[0, :])),  axis=0)
    maxs = np.max(   np.vstack((gbr1[1, :], gbr2[1, :])),  axis=0)
    return np.vstack( (mins, maxs) )

def inside_gabarits(pt, gbr): 
    #return all(pt >= gbr[0]) and all(pt <= gbr[1])     
    
    # !!! faster
    return (  (pt[0] >= gbr[0, 0]) and (pt[1] >= gbr[0, 1]) and (pt[2] >= gbr[0, 2]) and 
              (pt[0] <= gbr[1, 0]) and (pt[1] <= gbr[1, 1]) and (pt[2] <= gbr[1, 2])     )  

def outside_gabarits(pt, gbr): 
    #return any(pt < gbr[0]) or any(pt > gbr[1])     
    
    # !!! faster
    return (  (pt[0] < gbr[0, 0]) or (pt[1] < gbr[0, 1]) or (pt[2] < gbr[0, 2]) or 
              (pt[0] > gbr[1, 0]) or (pt[1] > gbr[1, 1]) or (pt[2] > gbr[1, 2])    )  


def testpoints(gbr, n): 
    c = gbr[0]
    k = gbr[1] - gbr[0]
    return np.random.rand(n, 3)*k + c
    
#%%

def get_coord_indexes(axes_code):
    axes_dict = {None: (0, 1), 'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1),
                               'YX': (1, 0),' ZX': (2, 0), 'YZ': (1, 2)}
    return axes_dict[axes_code]

def plot_point(pt, axes_code=None, **kwargs): 
    X, Y = get_coord_indexes(axes_code)
    xx = [pt[X]] 
    yy = [pt[Y]] 
    return plt.plot(xx, yy, 'o', **kwargs) # ms, markersize
    

def plot_segm(p0, p1, axes_code=None, **kwargs):
    X, Y = get_coord_indexes(axes_code)
    return plt.plot([  p0[X], p1[X]  ], 
                    [  p0[Y], p1[Y]  ], **kwargs)

def plot_polygon(points, axes_code=None, **kwargs):
    X, Y = get_coord_indexes(axes_code)

    xx = [pt[X] for pt in points] 
    xx.append(points[0][X])

    yy = [pt[Y] for pt in points] 
    yy.append(points[0][Y])

    return plt.plot(xx, yy, **kwargs)

