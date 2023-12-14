# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 08:59:14 2023

@author: reonid


# from collections import Iterable # for Python < 3.3
from collections.abc import Iterable   
if isinstance(the_element, Iterable): 
    pass
    
"""


import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from fastinterp import FastGridInterpolator3D, FastGridInterpolator2D, TestGridInterpolator


class StopWatch:
    def __init__(self, title=''): 
        self.title = title
        self.t0 = None
        self.t1 = None
        
    def __enter__(self): 
        self.t0 = time.time()
        return self
    
    def __exit__(self, exp_type, exp_value, traceback): 
        self.t1 = time.time()
        if exp_type is None: 
            print(self.title + ': dt = %.2f' % (self.t1 - self.t0))
        else: 
            print(self.title + ': error')
        
        # return True 
        return False # !!! don't suppress exception 

        
xx = np.linspace(-1.0,   10.0, 100)
yy = np.linspace(-15.8, 126.0, 120)
zz = np.linspace(-30.0,  30.0, 130)

x3d, y3d, z3d = np.meshgrid(xx, yy, zz, indexing='ij')  # !!! indexing='ij' - else it swaps x and y

#vv = np.ones((len(xx), len(yy), len(zz)))
v3d = np.sin(x3d*1.7 + y3d*0.18 + z3d*0.98)


fastintp = FastGridInterpolator3D ( (xx, yy, zz), v3d, fill_value=0.0)
testintp = TestGridInterpolator ( (xx, yy, zz), v3d, fill_value=0.0)
regintp  = RegularGridInterpolator( (xx, yy, zz), v3d, fill_value=0.0, bounds_error=False)

#%%

pt = np.array([0.2, 0.4, 0.5])
v = fastintp(pt)
_v = regintp(pt)
print('reg:   ', v)
print('fast: ',  _v)

if not np.isclose(v, _v): 
    raise Exception('Test failed 0')


#%%

with StopWatch('reg'): 
    for i in range(10000): 
        v = regintp(pt)

with StopWatch('fast'): 
    for i in range(10000): 
        _v = fastintp(pt)

with StopWatch('test'): 
    for i in range(10000): 
        __v = testintp(pt)

#%% 3d 

pts = pt
for i in range(1000): 
    pts = np.vstack( (pts, pt + 0.01*i) )

with StopWatch('reg[]'): 
    for i in range(1000): 
        vv = regintp(pts)

with StopWatch('fast[]'): 
    for i in range(1000): 
        _vv = fastintp(pts)

with StopWatch('test[]'): 
    for i in range(1000): 
        __vv = testintp(pts)


if not np.isclose( np.sum(vv - _vv), 0.0): 
    raise Exception('Test failed 1')

if not np.isclose( np.sum(vv - __vv), 0.0): 
    raise Exception('Test failed 2')

#%% Visual test, containing points outside domain

vv  = [fastintp([x*1.0001, 0.11, 22.0]) for x in xx]
_vv = [ regintp([x*1.0001, 0.11, 22.0]) for x in xx]
plt.plot(xx, vv)
plt.plot(xx, _vv)

#%% 2d

fastintp2d = FastGridInterpolator2D ( (xx, yy), v3d[:, :, 0], fill_value=0.0)
testintp2d = TestGridInterpolator   ( (xx, yy), v3d[:, :, 0], fill_value=0.0)
regintp2d  = RegularGridInterpolator( (xx, yy), v3d[:, :, 0], fill_value=0.0, bounds_error=False)

v   = regintp2d(pt[0:2])
_v  = fastintp2d(pt[0:2])
__v = fastintp2d(pt[0:2])

if not np.isclose(v,  _v): 
    raise Exception('Test failed 3')

if not np.isclose(v,  __v): 
    raise Exception('Test failed 4')
    
#%% 

with StopWatch('test'): 
    for i in range(10000): 
        __v = testintp2d(pt[0:2])

with StopWatch('reg2d[]'): 
    for i in range(1000): 
        vv = regintp2d(pts[:, 0:2])

with StopWatch('fast2d[]'): 
    for i in range(1000): 
        _vv = fastintp2d(pts[:, 0:2])

with StopWatch('test2d[]'): 
    for i in range(1000): 
        __vv = testintp2d(pts[:, 0:2])

if not np.isclose( np.sum(vv - __vv), 0.0): 
    raise Exception('Test failed 5')

if not np.isclose( np.sum(_vv - __vv), 0.0): 
    raise Exception('Test failed 6')
