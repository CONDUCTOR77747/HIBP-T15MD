# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:44:43 2023

@author: reonid
"""

import numpy as np
from scipy.ndimage import convolve

from .geom import indexing

# %%

_1_ = indexing[1:-1]

kernel1D = np.array([1.0,  0.0,  1.0]) # just for generality

kernel2D = np.array([[0.0,  1.0,  0.0],
                     [ 1.0,  0.0,  1.0],
                     [ 0.0,  1.0,  0.0]], dtype=np.float64)

kernel3D = np.zeros((3, 3, 3), dtype=np.float64)
kernel3D[-1, 1, 1] = 1.0; kernel3D[0, 1, 1] = 1.0
kernel3D[ 1,-1, 1] = 1.0; kernel3D[1, 0, 1] = 1.0
kernel3D[ 1, 1,-1] = 1.0; kernel3D[1, 1, 0] = 1.0 

def mid_indexing(U): 
    if len(U.shape) == 1: 
        return U.shape[0]//2
    elif len(U.shape) == 2: 
        return indexing[U.shape[0]//2, U.shape[1]//2]
    elif len(U.shape) == 3: 
        return indexing[U.shape[0]//2, U.shape[1]//2, U.shape[2]//2]

def get_kernel(U): 
    if len(U.shape) == 1: 
        return kernel1D
    elif len(U.shape) == 2: 
        return kernel2D
    elif len(U.shape) == 3: 
        return kernel3D


class LaplaceBoundaryConditions: 
    def __init__(self, masks_voltages):
        self.masks_voltages = masks_voltages
        
    def reset(self, U): 
        for mask, voltage in self.masks_voltages: 
            U[mask] = voltage     # include boundary conditions U[edge_flag] = 0.0
            # !!! here can use indexing as mask

        # Neumann boundary condition on left and right sides
        # ??? is it equivalent to 'nearest' ??? yes
        # U[   0,  _1_,  _1_ ] =  U[   1,  _1_,  _1_ ]  
        # U[  -1,  _1_,  _1_ ] =  U[  -2,  _1_,  _1_ ]

        return U # !!!

def create_edge_mask3D(U, x_edge, y_edge, z_edge): # x_edge=[], y_edge=[0,-1], z_edge=[0,-1]
    edge_mask = np.full_like(U, False, dtype=bool)
    edge_mask[x_edge, :, :] = True 
    edge_mask[:, y_edge, :] = True
    edge_mask[:, :, z_edge] = True

    return edge_mask

# indexing[:, [0, -1], :], indexing[:, :, [0, -1]]

#%%

#def sum_of_subarrays(U, result_indexing, *args): 
#    result = np.zeros_like( U[result_indexing] )
#    for idx in args: 
#        result += U[idx]
#    return result    
#
#
#def _solve(U, boundary_conditions, test_indexing=None, eps=1e-5): 
#    ''' solve Laplace equation
#    '''
#    # may be not best naming, but visually distinguishable
#    _xyz_ = indexing[1:-1,  1:-1,  1:-1 ] # _1_,  _1_,  _1_
#    _mid_ = indexing[U.shape[0]//2, U.shape[1]//2, U.shape[2]//2]
#    x__, __x = indexing[0:-2,  1:-1,  1:-1], indexing[2:  ,  1:-1,  1:-1]
#    y__, __y = indexing[1:-1,  0:-2,  1:-1], indexing[1:-1,  2:  ,  1:-1]
#    z__, __z = indexing[1:-1,  1:-1,  0:-2], indexing[1:-1,  1:-1,  2:  ]
#    
#    if test_indexing is None: 
#        test_indexing = _mid_
#
#    test_U0 = np.copy( U[test_indexing] ) 
#    test_U1 = np.full_like(test_U0, 1e3)
#    step = 0
#
#    #while abs(U1[_mid_] - U0[_mid_]) > eps:  # np.abs  
#    #while np.amax( np.abs(U1-U0) ) > eps: # ??? amax ???
#    while np.max( np.abs(test_U1 - test_U0) ) > eps:
#        step += 1 
#        test_U0 = np.copy( U[test_indexing] )   # np.copy(U)
#        boundary_conditions.reset(U)       # apply initial conditions at every time step
#        
#        U[_xyz_] = sum_of_subarrays(U, _xyz_,      x__, __x, y__, __y, z__, __z)*0.16666666666666666
#
#        if step > 1000:  # wait until potential spreads to the center point        
#            test_U1 = np.copy( U[test_indexing] )
#
#    print('Total number of steps = {}'.format(step))
#    #print('sum(U) = ', np.sum(U)) 
#    boundary_conditions.reset(U) 
#    return U


def laplace_solve(U, boundary_conditions, test_indexing=None, eps=1e-5):

    kernel = get_kernel(U)
    kernel = kernel/np.sum(kernel) 

    if test_indexing is None: 
        test_indexing = mid_indexing(U)

    _U = U.copy()

    step = 0
    skip_steps = max(U.shape)*2

    while True:
        step += 1
        _U, U = U, _U
        boundary_conditions.reset(_U)      # always _U -> U
        convolve(_U, kernel, mode='nearest', output=U)

        if (step < skip_steps): 
            continue
        
        if step % 20 == 0: # Check can be expensive
            if (np.max( np.abs(U[test_indexing] - _U[test_indexing]) ) < eps): 
                break

    print('Total number of steps = {}'.format(step))

    boundary_conditions.reset(U) 
    return U


#%%

def calcE(U, grid_step, plates_masks): 
    # explicitely for 3D and 2D
    if len(U.shape) == 3: 
        Ex, Ey, Ez = np.gradient(-1.0*U, grid_step)
        # set zero E in the cells corresponding to plates
        for mask in plates_masks: 
            Ex[mask], Ey[mask], Ez[mask] = 0.0, 0.0, 0.0
        
        return Ex, Ey, Ez

    elif len(U.shape) == 2: 
        Ex, Ey = np.gradient(-1.0*U, grid_step)
        # set zero E in the cells corresponding to plates
        for mask in plates_masks: 
            Ex[mask], Ey[mask] = 0.0, 0.0
        
        return Ex, Ey
    else: 
        return None # ???

#    Both for 2D and 3D
#    EE = np.gradient(-1.0*U, grid_step)   
#    # set zero E in the cells corresponding to plates
#    for mask in plates_masks: 
#        for E_ in EE: 
#          E_[mask] = 0.0
#    
#    return EE

