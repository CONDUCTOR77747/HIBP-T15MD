# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:57:08 2022

Hilbert-Huang transform test

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import hilbert #, chirp

from ..xysig import XYSignal
from ..hibpcarp import sampling_rate

MAX_EMD_ITER = 100
MAX_EMD_COMP = 100


#%%

def loc_min_max_mask(yy): 
    yy_m = np.roll(yy, -1)
    yy_p = np.roll(yy, 1)
    
    dd_m = yy - yy_m
    dd_p = yy - yy_p
    
    min_mask = (dd_m <= 0.0) & (dd_p <= 0.0)
    max_mask = (dd_m >= 0.0) & (dd_p >= 0.0)
    return min_mask, max_mask

def loc_min_max_arr(xx, yy): 
    min_mask, max_mask = loc_min_max_mask(yy)
    min_xx = xx[min_mask]
    min_yy = yy[min_mask]
    
    max_xx = xx[max_mask]
    max_yy = yy[max_mask]
    return min_xx, min_yy, max_xx, max_yy

def loc_min_max_spline(xx, yy): 
    min_xx, min_yy, max_xx, max_yy = loc_min_max_arr(xx, yy)

    min_tck = interpolate.splrep(min_xx, min_yy)
    max_tck = interpolate.splrep(max_xx, max_yy)
    
    def minspline(x): 
        return interpolate.splev(x, min_tck)

    def maxspline(x): 
        return interpolate.splev(x, max_tck)
    
    def meanspline(x): 
        return 0.5*(interpolate.splev(x, max_tck) + interpolate.splev(x, min_tck))
    
    return minspline, maxspline, meanspline

#%%

def extract_single_emd(xx, yy, stopfunc): 
    hh = yy    
    for i in range(MAX_EMD_ITER): 
        minsplinefunc, maxsplinefunc, meansplinefunc = loc_min_max_spline(xx, hh)
        mean_hh = meansplinefunc(xx)
        hh = hh - mean_hh
        if stopfunc(hh): 
            break    
    rr = yy - hh    
    return hh, rr

def decompose_into_emd(xx, yy, stopfunc): 
    result = []
    hh = yy
    rr = yy

    for i in range(MAX_EMD_COMP): 
        try: 
            hh, rr = extract_single_emd(xx, rr, stopfunc)
            result.append(hh)
        except: 
            break
    return result, rr

#%%


class HilbertHuang: 
    def __init__(self, xx, yy, stopfunc=None): 
        self.x = xx
        self.y = yy
        fs = sampling_rate(xx)
        
        d = np.max(yy) - np.min(yy)
        if stopfunc is None: 
            stopfunc = lambda g: np.max(g) < d*0.002
        
        self.emd = []
        self.ampl = []
        self.phase = []
        self.freq = []

        _emd, self.r = decompose_into_emd(xx, yy, stopfunc)

        for _mode in _emd: 
            analytic_signal = hilbert(_mode)
            # y = np.real(analytic_signal)
            # hy = np.imag(analytic_signal)        
            self.emd.append( analytic_signal )
            self.ampl.append( np.abs(analytic_signal) )

            ph = np.unwrap(np.angle(analytic_signal))
            self.phase.append(ph)

            f =  (np.diff(ph) / (2.0*np.pi) * fs) 
            f = np.append(f, 0.0)   # ??? resample or just shift ???
            self.freq.append(f)
    
    def quantity(self, name, idx): 
        if idx >= len(self.emd): 
            return None
        
        if name == "ampl": 
            yy = self.ampl[idx]
        elif name == "freq": 
            yy = self.freq[idx]
        elif name == "phase": 
            yy = self.phase[idx]
        else: 
            return None
        
        return XYSignal((self.x, yy))
        

#%%

if __name__ == "__main__": 
    xx = np.linspace(0.0, 1000.0, 10000)
    #yy = np.sin(xx*10.0) + np.sin(xx*100.0)
    #yy = 0*np.sin(xx*10.0) + np.sin(xx*100.0)
    yy = np.sin(xx*(1.0- xx*0.0003) ) + np.sin(xx*(0.10 - xx*0.00003)) + (0.2 + xx*0.000001)*np.sin(xx*(0.33 + xx*0.000001))
    
    plt.plot(xx, yy)
    plt.figure()
    hh = HilbertHuang(xx, yy)
    plt.plot(hh.x, hh.freq[0])
    plt.plot(hh.x, hh.freq[1])
    plt.plot(hh.x, hh.freq[2])

    #plt.plot(hh.x, hh.phase[0])
    #plt.plot(hh.x, hh.phase[1])
    #plt.plot(hh.x, hh.phase[2])
    
    #plt.plot(hh.x, hh.ampl[0])
    #plt.plot(hh.x, hh.ampl[1])
    #plt.plot(hh.x, hh.ampl[2])



