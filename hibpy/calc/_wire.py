# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:46:36 2023

@author: reonid
"""

import numpy as np

from joblib import Parallel, delayed
import multiprocessing as mp

import matplotlib.pyplot as plt
import math as math
#import types as types

from .ring import Ring, magneticFdOnRingCenter, _magneticFdOfRing3D #, FilamentGroup
from .geomfunc import pt3D, vec3D, line_array, vNorm


SI_mu0 = 4*np.pi*1e-7

sin = np.sin
cos = np.cos  # ??? math
atan2 = math.atan2

sin = math.sin
cos = math.cos  # ??? math

EPS_WIRE = 0.05 # m  # ???

def calc_Bpoint(r, IdL, r_elem):
    r2 = r - r_elem
    r25 = np.linalg.norm(r2, axis=1)
    # create a mask to exclude points close to wire
    mask = (r25 < EPS_WIRE)  # 0.6 is ok for plasma current field
    r25 = r25**3
    r3 = r2 / r25 [:, np.newaxis]

    cr = np.cross(IdL, r3)
    cr[mask] = [0., 0., 0.]

    # claculate sum of contributions from all current elements
    s = np.sum(cr, axis=0)
    return s #*1e-7

def elem_biot_savart(r, IdL, r_elem, eps):
    dr = r - r_elem
    abs_dr = vNorm(dr)

    if abs_dr > eps: 
        b = np.cross(IdL, dr) / abs_dr**3
    else: 
        b = 0.0

    return b*1e-7

def biot_savart(points, wires):
    '''
    calculate the magnetic field generated by currents flowing through wires
    :param wires: list of wire objects
    :param points: numpy array with x,y,z coordinates of n points
    :return: numpy array of n vectors representing the B field at given points
    '''
    if len(wires) == 0:  return np.nan
    if len(points) == 0: return np.nan

    c = 0
    # generate list of IdL and r1 vectors from all wires
    for w in wires:
        c += 1
        _IdL, _r1 = w.IdL_r1()
#        print('wire {} has {} segments'.format(c, len(_IdL)))
        if c == 1:
            IdL = _IdL
            r1 = _r1
        else:
            IdL = np.vstack((IdL, _IdL))
            r1 = np.vstack((r1, _r1))

    # now we have all segment vectors multiplied by the flowing current in IdL
    # and all vectors to the central points of the segments in r1
    
    # calculate vector B*1e7 for each point in space
    # calculate B at each point r

    # single processor
    # B = np.array([calc_Bpoint(r, IdL, r1) * 1e-7 for r in points])

    # multiprocessing
    n_workers = mp.cpu_count() - 1
    s = Parallel(n_jobs=n_workers)(delayed(calc_Bpoint)(r, IdL, r1) for r in points)
    B = np.array(s)*1e-7

    return B



class Wire: 
    def __init__(self, path, I): 
        self.I = I
        self.path = path
        self.close_path()       

    def close_path(self):
        pass
        if not np.all(self.path[-1] == self.path[0]): 
            self.path = np.append(self.path, self.path[0:1], axis=0)

    @property
    def IdL_r1(self):
        '''
        calculate discretized path elements dL and their center point r1
        :return: numpy array with I * dL vectors, numpy array of r1 vectors 
        (center point of element dL)
        '''
        N = len(self.path)
        if N < 2:
            raise Exception("discretized path must have at least two points")

        IdL = np.array([self.path[c+1] - self.path[c] for c in range(N-1)]) * self.I
        r1 = np.array([(self.path[c+1] + self.path[c])*0.5 for c in range(N-1)])

        return IdL, r1

    
    def calcB(self, r): 
        b = np.zeros(3)
        IdL, r1 = self.IdL_r1
#        for idl, rw in zip(IdL, r1): 
#            b += calc_Bpoint(r.reshape(1, 3), idl, rw)
#        return b*1e-7
        for idl, rw in zip(IdL, r1): 
            b += elem_biot_savart(r, idl, rw, EPS_WIRE) #calc_Bpoint(r.reshape(1, 3), idl, rw)
        return b
    
    def calcB_(self, r): 
        b = np.zeros(3)
        IdL, r1 = self.IdL_r1
        b = biot_savart(r.reshape(1, 3), [self])
        return b
    
    
    def plot(self): 
        xx = self.path[:, 0]
        yy = self.path[:, 1]
        plt.plot(xx, yy)


