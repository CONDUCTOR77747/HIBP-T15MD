# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:15:50 2023

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt

from hibpcalc.geom import (pt3D, vec3D, size3D, 
                  identMx, invMx, rotateMx, xwardRotateMx, xScaleMx, skewMx, 
                  plot_point, plot_polygon, 
                  calc_gabarits, join_gabarits, testpoints)

import hibpcalc.geomfunc as gf
import hibpcalc.fields as fields
from hibpcalc.box import Box3D, box3D, Group3D, FastGroup3D, BoundlessGroup3D
import hibpcalc.misc as misc

def new_plate_flags(range_x, range_y, range_z, U,
                plts_geom, plts_angles, plts_center):
    '''
    calculate plates cooedinates and boolean arrays for plates
    '''
    length, width, thick, gap, l_sw = plts_geom
    gamma, alpha_sw = plts_angles
    
    thick *= 1.0005
    gap /= 1.0001
#    upper = Box3D(length - l_sw, thick, width).trans( vec3D(l_sw*0.5,  (gap+thick)*0.5, 0) )
#    lower = Box3D(length - l_sw, thick, width).trans( vec3D(l_sw*0.5, -(gap+thick)*0.5, 0) )
#    
#    upper_ = Box3D(l_sw, thick, width).trans( vec3D(0.5*(-length+l_sw),  (gap+thick)*0.5, 0) )
#    lower_ = Box3D(l_sw, thick, width).trans( vec3D(0.5*(-length+l_sw), -(gap+thick)*0.5, 0) )

    
    upper  = box3D( (-length*0.5+l_sw, +length*0.5), (-thick*0.5, +thick*0.5), (-width*0.5, +width*0.5)).trans( vec3D(0,  (gap+thick)*0.5, 0) ) 
    lower  = box3D( (-length*0.5+l_sw, +length*0.5), (-thick*0.5, +thick*0.5), (-width*0.5, +width*0.5)).trans( vec3D(0, -(gap+thick)*0.5, 0) )
    
    upper_ = box3D( (-length*0.5, -length*0.5+l_sw), (-thick*0.5, +thick*0.5), (-width*0.5, +width*0.5)).trans( vec3D(0,  (gap+thick)*0.5, 0) ) 
    lower_ = box3D( (-length*0.5, -length*0.5+l_sw), (-thick*0.5, +thick*0.5), (-width*0.5, +width*0.5)).trans( vec3D(0, -(gap+thick)*0.5, 0) ) 
   
    skew_plane = pt3D(-length/2.0 + l_sw, 0, 0), vec3D(1, 0, 0)
    upper_.rotate_with_skew(skew_plane, -np.deg2rad(alpha_sw) )
    lower_.rotate_with_skew(skew_plane,  np.deg2rad(alpha_sw) )

    uppergroup = FastGroup3D([upper, upper_])
    lowergroup = FastGroup3D([lower, lower_])
    framegroup = BoundlessGroup3D()  # ???

    fullgroup = BoundlessGroup3D([uppergroup, lowergroup, framegroup]) 
    fullgroup.rotate(pt3D(0, 0, 0), vec3D(1, 0, 0), np.deg2rad(gamma)).trans(plts_center)

    upper_plate_flag = np.full_like(U, False, dtype=bool)
    lower_plate_flag = np.full_like(U, False, dtype=bool)

    pt = np.zeros((3,))
    for i, x in enumerate(range_x):
        for j, y in enumerate(range_y):
            for k, z in enumerate(range_z):
                #pt = np.array([x, y, z])
                pt[0], pt[1], pt[2] = x, y, z   # !!! faster 

                upper_plate_flag[i, j, k] = uppergroup.contains_pt( pt )
                lower_plate_flag[i, j, k] = lowergroup.contains_pt( pt )

    flared = abs(alpha_sw) > 1e-2    

    return uppergroup, lowergroup, upper_plate_flag, lower_plate_flag
    #return (points_from_platesgroup(uppergroup, flared, False), 
    #         points_from_platesgroup(lowergroup, flared, True), 
    #         upper_plate_flag, lower_plate_flag)

def plot_slice(flags3D, z_idx, xx, yy, d=0.0, **kwargs): 
    flags2D = flags3D[:, :, z_idx]
    for i in range(flags2D.shape[0]): 
        for j in range(flags2D.shape[1]): 
            if flags2D[i, j]: 
                plot_point((xx[i]+d, yy[j]+d), **kwargs)

def plot_points(points, **kwargs): 
    xx = points[:, 0]
    yy = points[:, 1]
    plt.plot(xx, yy, **kwargs)
#    for i, (x, y) in enumerate(zip(xx, yy)): 
#        plt.text(x, y-i*0.0004, str(i))

def points_from_platesboxes(box, box_, flared, lower): 
    p = box.points()
    p_ = box_.points()
    if flared: 
        result = [ p_[0], p_[1], p_[5], p_[4],  p[4],  p[0],  
                   p_[3], p_[2], p_[6], p_[7],  p[7], p[3]  ]
        if lower: 
            result = result[6:] + result[0:6]    
    else: 
        result = [ p_[1], p_[5], p[4],  p[0],  p_[2], p_[6], p[7], p[3]  ]
        if lower: 
            result = result[4:] + result[0:4]
    
    return result    
        
def points_from_platesgroup(grp, flared, lower): 
    #Y = 1
    box = grp.elements()[0]
    box_ = grp.elements()[1]
    #flared = abs(box.points()[1][Y] - box_.points()[1][Y]) > 1e-6
    return points_from_platesboxes(box, box_, flared, lower)

def eq_point_lists(pts1, pts2): 
    for p1, p2 in zip(pts1, pts2): 
        if not all( np.isclose(p1, p2) ): 
            return False
    return True    
    

#%%

plts_name = 'A2'

# define voltages [Volts]
Uupper_plate = 0.
Ulower_plate = 1e3

plts_center = np.array([0., 0., 0.])  # plates center
gamma = 0.  # gamma = 0. for A-plates, and -90. for B-plates

# plts_center = np.array([10.5, 03., 0.])  # plates center
# gamma = -90.  # gamma = 0. for A-plates, and -90. for B-plates

drad = np.pi/180. # convert degrees to radians

# define plates geometry 'A2'
beamline = 'prim'
length = 0.35  # along X [m]
width = 0.1  # along Z [m]
thick = 0.005  # [m]
gap = 0.05  # distance between plates along Y [m]
alpha_sw = 10.0  # sweep angle [deg] for flared plates
l_sw = 0.15  # length of a flared part

# set plates geometry
plts_geom = np.array([length, width, thick, gap, l_sw])
plts_angles = np.array([gamma, alpha_sw])

# Create mesh grid
# lengths of the edges of the domain [m]
r = np.array([length, gap, width])
r = gf.rotate(r, axis=(0, 0, 1), deg=alpha_sw)
r = gf.rotate(r, axis=(1, 0, 0), deg=gamma)
r = abs(r)
border_x = round(2*r[0], 2)
border_y = round(2*r[1], 2)
border_z = round(2*r[2], 2)
delta = thick/2  # space step

range_x = np.arange(-border_x/2., border_x/2., delta) + plts_center[0]
range_y = np.arange(-border_y/2., border_y/2., delta) + plts_center[1]
range_z = np.arange(-border_z/2., border_z/2., delta) + plts_center[2]

x, y, z = np.meshgrid(range_x, range_y, range_z, indexing='ij')  # [X ,Y, Z]
mx = range_x.shape[0]
my = range_y.shape[0]
mz = range_z.shape[0]

# define mask for edge elements
edge_flag = np.full_like(x, False, dtype=bool)
edge_list = [0]  # indexes of edge elements
# edge_flag[edge_list, :, :] = True
edge_flag[:, edge_list, :] = True
edge_flag[:, :, edge_list] = True

# array for electric potential
U = np.zeros((mx, my, mz))

#%%

with misc.StopWatch('old'): 
    UP, LP, upper_plate_flag, lower_plate_flag = fields.plate_flags(
            range_x, range_y, range_z, U, plts_geom, plts_angles, plts_center)

# 23% slower (initially 300%)
with misc.StopWatch('new'): 
    _UP, _LP, _upper_plate_flag, _lower_plate_flag = new_plate_flags(
            range_x, range_y, range_z, U, plts_geom, plts_angles, plts_center)

#%%

plt.figure()

z_idx = upper_plate_flag.shape[2] // 2
plot_slice(_upper_plate_flag, z_idx, range_x, range_y, ms=2, color='r')
plot_slice(_lower_plate_flag, z_idx, range_x, range_y, ms=2, color='b')


#plot_slice(_upper_plate_flag, z_idx, range_x, range_y, d=0.0000, ms=2, color='g')
#plot_slice(_lower_plate_flag, z_idx, range_x, range_y, d=0.0000, ms=2, color='orange')

_UP.plot()
_LP.plot()

plot_points(UP)
plot_points(LP)


shp = upper_plate_flag.shape
N = shp[0]*shp[1]*shp[2]
_n = np.sum( upper_plate_flag ^ _upper_plate_flag)
print('Mismatch old/new: ')
print('   ', _n, '  from ', N, ' (%.3f %%)' % (_n/N*100.0)    )

if not eq_point_lists(UP, points_from_platesgroup(_UP, True, False)): 
    print('UP test failed')

if not eq_point_lists(LP, points_from_platesgroup(_LP, True, True)): 
    print('LP test failed')