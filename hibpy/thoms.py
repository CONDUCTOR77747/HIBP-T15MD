# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:36:14 2019

@author: reonid

"""

import numpy as np
#from scipy.optimize import curve_fit
from scipy import integrate

from .bcache import loadThomsonSignal
from .xysig import XYSignal #, SignalError
from .hibpsig import signal
from .fit.hollow import van_milligen_func #, hollow_func
from .cienptas import update_default_kwargs

_thomson_navg = {}

_thomson_fit_func_for_calib = van_milligen_func
'''
cached live-averaged densities for TS instants
'''

class ThomsonSignal(XYSignal): 
    def __init__(self, xy=(None, None), copy=False):
        super().__init__(xy, copy)
        self.time = None
        self.lineavg = None
        self.smoothed = None
        self._sigma = None

    def get_experimental_lineavg_dens(self): 
        ne = _thomson_navg.get(self.shot, None)
        if ne is None: 
            try: 
                dens = signal(self.shot, 'DENCM0_{avg555n11}', device='tjii') 
            except: 
                dens = signal(self.shot, 'Densidad2_{avg55n11}') 
           
            ne = dens.valueat(self.time)
        return ne
        
    def calibrate(self, fitfunc=None): 
        if self.name not in ['Ne', 'ne']: 
            return
        
        if fitfunc is None: 
            fitfunc = _thomson_fit_func_for_calib

        self.fit(fitfunc)
        
        n_exp = self.get_experimental_lineavg_dens()

        n_ = self.lineavg
        self.y *= n_exp/n_
        self.lineavg *= n_exp/n_
        self.smoothed.y *= n_exp/n_
        self.smoothed.lineavg *= n_exp/n_

    def recalibrate(self, smoothed_prof):
        if self.name not in ['Ne', 'ne']: 
            return
        
        n_exp = self.get_experimental_lineavg_dens()

        n_ = smoothed_prof.lineavg

        self.y *= n_exp/n_
        self.lineavg *= n_exp/n_
        self.smoothed = smoothed_prof
        
        smoothed_prof.y *= n_exp/n_
        smoothed_prof.lineavg *= n_exp/n_
        
    @classmethod 
    def fromcache(cls, shot, name, calib=False): 
        if calib and (name == 'Pe')or(name == 'PerfilPe'): 
            Te = cls.fromcache(shot, 'Te')
            ne = cls.fromcache(shot, 'ne', calib=True)
            result = Te
            result.y = result.y*ne.y
            return result
        
        xy, t = loadThomsonSignal(shot, name)
        prof = cls.fromdata(shot, name, xy)
        prof.time = t
        if calib: 
            prof.calibrate()
        return prof

    @classmethod 
    def fromdata(cls, shot, name, xy): 
        prof = cls(xy)
        prof.shot = shot
        prof.name = name
        return prof
        
    @classmethod 
    def merged_profile(cls, shots, name, calib=False): 
        xx, yy = None, None
        for n in shots: 
            prof = cls.fromcache(n, name, calib)
            xx = prof.x if xx is None else np.hstack((xx, prof.x))
            yy = prof.y if yy is None else np.hstack((yy, prof.y))

        ordered_indices = xx.argsort()
        xx = xx[ordered_indices]
        yy = yy[ordered_indices]

        return cls.fromdata(shots, name, (xx, yy))

    def add_prof(self, prof): 
        self.x = np.hstack((self.x, prof.x))
        self.y = np.hstack((self.y, prof.y))
        
        if (self._sigma is not None)and(prof._sigma is not None): 
            self._sigma = np.hstack((self._sigma, prof._sigma))
        else:     
            self._sigma = None

        ordered_indices = self.x.argsort()
        self.x = self.x[ordered_indices]
        self.y = self.y[ordered_indices]
        
        if self._sigma is not None: 
            self._sigma = self._sigma[ordered_indices]

        
    def fit(self, func, mesh=None, **kwargs): 
        '''
        popt, pcov = curve_fit(func, self.x, self.y, **kwargs)
        xx = np.linspace(-1.0, 1.0, 100) if mesh is None else mesh 
        yy = func(xx, *popt)
        result = ThomsonSignal.fromdata(self.shot, self.name, (xx, yy))
        
        intg, err = integrate.quad(lambda x: func(x, *popt), -1, 1)
        result.lineavg = intg*0.5
        self.lineavg = result.lineavg
        self.smoothed = result
        result._fitopt = popt
        result._fitfunc = func
        return result
        '''
        
        #result = super().fit(func, mesh, sigma = self._sigma, **kwargs)
        result = super().fit(func, mesh, sigma = self._sigma, absolute_sigma=True, **kwargs)
        result.time = self.time
        
        intg, err = integrate.quad(lambda x: func(x, *result._fitopt), -1, 1)
        result.lineavg = intg*0.5
        self.lineavg = result.lineavg
        self.smoothed = result

        return result

    def __calc_lineavg(self): 
        fitopt = self.smoothed._fitopt if self.smoothed is not None else self._fitopt
        func = self._fitfunc
        intg, err = integrate.quad(lambda x: func(x, *fitopt), -1, 1)
        return intg*0.5
        

    @classmethod 
    def set_default_fit_func_for_calib(self, fitfunc): 
        global _thomson_fit_func_for_calib
        _thomson_fit_func_for_calib = fitfunc

    @classmethod 
    def set_experimental_thomson_densities(self, shot_dens): 
        global _thomson_navg
        _thomson_navg.update(shot_dens)

def thomson(shot, name, calib=False): 
    if isinstance(shot, list): 
        return ThomsonSignal.merged_profile(shot, name, calib)
    else: 
        return ThomsonSignal.fromcache(shot, name, calib)

#------------------------------------------------------------------------------
    
if __name__ == '__main__':
    pass

