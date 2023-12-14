# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:36:14 2019

@author: reonid

    
    
"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
import hollow

from cienptas import (update_default_kwargs, func_from_file, iter_ranges, 
                      expand_mask, narrow_mask, ispair, fit)

#%%

def update_default_kwargs_test(): 
    def f(**kwargs): 
        kwargs = update_default_kwargs(mlab.psd, kwargs, {'detrend':'mean', 'NFFT':2048, 'pad_to':4096, 'wrong_arg':'unused'})
        return kwargs
    
    kw = f()
    assert kw['NFFT'] == 2048
    assert kw['detrend'] == 'mean'
    assert kw['pad_to'] == 4096
    assert 'wrong_arg' not in kw.keys()
    
    kw = f(NFFT=512, detrend='linear')
    assert kw['NFFT'] == 512
    assert kw['detrend'] == 'linear'
    assert kw['pad_to'] == 4096
    assert 'wrong_arg' not in kw.keys()
    
    kw = f(NFFT=128, detrend='default', pad_to=1024, wrong_arg=3, unknown=None)
    assert kw['NFFT'] == 128
    assert kw['detrend'] == 'default'
    assert kw['pad_to'] == 1024
    assert 'wrong_arg' not in kw.keys()
    assert 'unknown' not in kw.keys()
    
    print('update_default_kwargs: OK')

#%%

'''
File for testing 
------------func.txt-------------
0   0.0   8   0 
1   0.1   7   0
2   0.4   6   0
3   0.9   5   1
4   1.6   4   1
5   2.5   3   1
6   3.6   2   0
7   4.9   1   0
8   6.4   0   0
'''    

def func_from_file_test(): 
    f = func_from_file('func.txt', 0, 1, fill_value='const')
    
    #xx = np.linspace(-2, 10, 333)
    #plt.plot(xx, f(xx))

    assert np.isclose( f(-1), 0.0)
    assert np.isclose( f(10), 6.4)
    assert np.isclose( f(5), 2.5)
    assert np.isclose( f(5.5), 0.1*5.5**2, rtol=1e-2)

    f = func_from_file('func.txt', 0, 1, fill_value='const', skip_header=2, skip_footer=2)
    assert np.isclose( f(-1), 0.4)
    assert np.isclose( f(10), 3.6)
    assert np.isclose( f(5), 2.5)
    assert np.isclose( f(5.5), 0.1*5.5**2, rtol=1e-2)

    f = func_from_file('func.txt', fill_value=(-10, 10), skip_header=2, skip_footer=2)
    assert np.isclose( f(-1), -10)
    assert np.isclose( f(10), 10)
    assert np.isclose( f(5), 2.5)
    assert np.isclose( f(5.5), 0.1*5.5**2, rtol=1e-2)
    
    f = func_from_file('func.txt', 0, 0, fill_value='extrapolate', skip_header=2, skip_footer=2)
    assert np.isclose( f(-1), -1)
    assert np.isclose( f(5),   5)
    assert np.isclose( f(10), 10)    

    f = func_from_file('func.txt', 0, 2, fill_value='extrapolate', skip_header=2, skip_footer=2)
    assert np.isclose( f(0), 8)
    assert np.isclose( f(5.5), 8-5.5)    
    
    f = func_from_file('func.txt', xcol=0, ycol=2, fill_value='extrapolate', skip_header=2, skip_footer=2)
    assert np.isclose( f(0), 8)
    assert np.isclose( f(5.5), 8-5.5)    

    print('func_from_file: OK')
    
    
#%%    
    
def iter_ranges_test():
    #                0  1  2  3  4  5  6  7  8  9  10 11
    data = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    correct_ranges = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 12)]
    
    ranges = []
    for start, fin, ok, is1st, islast in iter_ranges(data, lambda x : x > 0.5): 
        ranges.append((start, fin))
    assert ranges == correct_ranges

    ranges = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x < 0.5): 
        ranges.append((start, fin))
    assert ranges == correct_ranges  # the same result

    # empty case
    data = np.array([])
    ranges = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        ranges.append((start, fin))
    assert ranges == []
    
    # homogeneous case
    data = np.array([1, 1, 1, 1])
    ranges = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        ranges.append((start, fin))
    assert ranges == [(0, 4)]

    data = np.array([1])
    ranges = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x < 0.5): 
        ranges.append((start, fin))
    assert ranges == [(0, 1)]

    # two ranges case
    data = np.array([0, 0, 0, 1, 1, 1])
    ranges = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x < 0.5): 
        ranges.append((start, fin))
    assert ranges == [(0, 3), (3, 6)]

    # Boundaries test
    data = np.array([0, 0, 0, 0])
    returnvalues = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 4, False, True, True)]

    data = np.array([1, 1, 1, 1, 1])
    returnvalues = []
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 5, True, True, True)]


    data = np.array([0, 0, 1, 1])
    returnvalues = []    
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 2, False, True, False), (2, 4, True, False, True)]

    data = np.array([1, 1, 0, 0])
    returnvalues = []    
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 2, True, True, False), (2, 4, False, False, True)]

    data = np.array([0, 0, 1, 1, 0, 0])
    returnvalues = []    
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 2, False, True, False), (2, 4, True, False, False), (4, 6, False, False, True)]

    data = np.array([1, 1, 0, 0, 1, 1])
    returnvalues = []    
    for start, fin, ok, is1st, islast  in iter_ranges(data, lambda x : x > 0.5): 
        returnvalues.append((start, fin, ok, is1st, islast))
    assert returnvalues == [(0, 2, True, True, False), (2, 4, False, False, False), (4, 6, True, False, True)]
    
    print('iter_ranges: OK')  


#%%

#  (A==B).all()    
#  np.array_equal(A,B)  # test if same shape, same elements values
#  np.array_equiv(A,B)  # test if broadcastable shape, same elements values
#  np.allclose(A,B,...)     

def expand_mask_test():
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype='int')
    m = expand_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype='int'))

    mask = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    m = expand_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([0, 0, 0, 0, 0, 0], dtype='int'))
    
    mask = np.array([1, 1, 1, 1, 1, 1], dtype='int')
    m = expand_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([1, 1, 1, 1, 1, 1], dtype='int'))

    mask = np.array([1, 0, 0, 0, 0, 0, 1], dtype='int')
    m = expand_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([1, 1, 1, 0, 1, 1, 1], dtype='int'))

    mask = np.array([1], dtype='int')
    m = expand_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([1], dtype='int'))

    mask = np.array([0], dtype='int')
    m = expand_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([0], dtype='int'))

    mask = np.array([], dtype='int')
    m = expand_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([], dtype='int'))

    print('expand_mask : OK')


def narrow_mask_test():
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype='int')
    m = narrow_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype='int'))

    mask = np.array([0, 0, 0, 0, 0, 0], dtype='int')
    m = narrow_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([0, 0, 0, 0, 0, 0], dtype='int'))
    
    mask = np.array([1, 1, 1, 1, 1, 1], dtype='int')
    m = narrow_mask(mask, 1, 1)
    assert np.array_equal(m, np.array([1, 1, 1, 1, 1, 1], dtype='int'))

    mask = np.array([0, 1, 1, 1, 1, 1, 0], dtype='int')
    m = narrow_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([0, 0, 0, 1, 0, 0, 0], dtype='int'))

    mask = np.array([1], dtype='int')
    m = narrow_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([1], dtype='int'))

    mask = np.array([0], dtype='int')
    m = narrow_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([0], dtype='int'))

    mask = np.array([], dtype='int')
    m = narrow_mask(mask, 2, 2)
    assert np.array_equal(m, np.array([], dtype='int'))

    print('narrow_mask : OK')


def ispair_test():
    assert not ispair(())
    assert not ispair((1, ))
    assert ispair((1, 2))
    assert not ispair((1, 2, 3))

    assert ispair((None, None))
    assert ispair((1, None))

    assert not ispair([])
    assert not ispair([1])
    assert not ispair([1, 2])
    assert not ispair([1, 2, 3])
    
    assert not ispair(1)
    assert not ispair(None)
    
    assert ispair((1, 2), dtype=int)
    assert ispair((1, 2), dtype=Number)
    assert not ispair((1, 2), dtype=float)
    assert ispair((1.0, 2.0), dtype=float)
    
    assert ispair((1, 2.0), dtype=(float, int))
    assert ispair((None, 2), dtype=(int, type(None)))    
    
    print('ispair : OK')

def fit_test(): 
    mesh = np.linspace(-0.99, 0.99, 100)
    x = np.array([-1.0, -0.6, -0.2, 0.3, 0.8, 1.0])
    y = np.array([-1.0, 0.6, 0.2, 0.3, 0.8, 1.0])
    #yy = fit(x, y, hollow.milligen_func, mesh)
    func = hollow.adapt4fit(hollow.hollow_func, scaling=True)
    func = hollow.hollow_func
    bounds=[(0.001, 0.07), (1.0, 10.0)]
    yy = fit(x, y, func, mesh=mesh, bounds=bounds)
    plt.plot(x, y, 'b+')
    plt.plot(mesh, yy)
    
if __name__ == '__main__':
    update_default_kwargs_test()
    func_from_file_test()
    iter_ranges_test()
    expand_mask_test()
    narrow_mask_test()
    ispair_test()
    fit_test()
    