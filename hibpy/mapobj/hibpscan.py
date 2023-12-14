# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:57:27 2020

@author: Administrator
"""

import sys
hibplib_path = 'G:\\myPy\\reonid-packages'
if hibplib_path not in sys.path: sys.path.append(hibplib_path)

from copy import deepcopy 
import numpy as np
import matplotlib.pyplot as plt
#from scipy import integrate
from scipy import diff

from ..cienptas import iter_ranges

from ..xysig import XYSignal
#from ..hibpsig import signal, Signal, config_loader


scan_signames = ['phi', 'rho', 'dens']  # 'itot'
scan_rhomesh = np.linspace(-1.05, 1.05, 200)

class AbstractHibpLoader: 
    def load(self, shot, alias):
        '''
        Should support aliases 'uscan', 'phi', 'itot', 'rho', 'dens'
        '''
        pass 

class PhiScan: 
    def __init__(self, phi, rho, dens): 
        self.phi = phi
        self.rho = rho
        self.dens = dens
        self.dn = None
        
        if self.phi is not None: 
            self.mean_dens = np.mean(dens.y)
            self._rearrange(scan_rhomesh)

    def _signals(self):
        for signame in scan_signames: 
            yield self.__dict__[signame]
        
    def _rearrange(self, rmesh):
        sortmap = self.rho.y.argsort()
        
        for sig in self._signals():
            sig.y = sig.y[sortmap]
            
        for sig in self._signals():
            sig.x = self.rho.y

        for sig in self._signals():
            sig.resample_as(rmesh)
        
        self.phi = self.phi.y
        self.rho = self.rho.y
        self.dens = self.dens.y

    @classmethod
    def emptyscan(cls): 
        result = cls(None, None, None)
        result.rho = deepcopy(scan_rhomesh)
        result.phi = np.zeros_like(result.rho)
        result.dens = np.zeros_like(result.rho)
        return result
        
    @classmethod
    def from_slice(cls, phi, rho, dens, slc): 
        phi = phi.get_obj_indexing(slc)
        rho = rho.get_obj_indexing(slc)
        dens = dens.get_obj_indexing(slc)
        
        scan = PhiScan(phi, rho, dens)
        return scan

    @classmethod
    def mean(cls, scans): 
        result = cls.emptyscan()
        cnt = 0
        for sc in scans: 
           result.phi += sc.phi
           result.dens += sc.dens
           cnt +=1
           
        result.phi /= cnt
        result.dens /= cnt
        return result   
        
    @classmethod
    def load_scans(cls, shot, loader): 
        uscan = loader.load(shot, 'uscan')
        phi = loader.load(shot, 'phi')
        rho = loader.load(shot, 'rho')
        dens = loader.load(shot, 'dens')

        du = deepcopy(uscan)
        du.y = diff(du.y)
        du.x = du.x[0:-1]

        du.resample_as(phi)
        rho.resample_as(phi)
        dens.resample_as(phi)
        
        scans = []
            
        for start, fin, ok, is1st, islast  in iter_ranges(du.y, lambda x: x > 0.0):
            scan = PhiScan.from_slice(phi, rho, dens, slice(start, fin))
            scans.append(scan)

        return scans

    def __mod__(self, other): # %
        return self.normdiff(other)
        
    def normdiff(self, other): 
        dp = self.phi - other.phi
        dn = self.dens - other.dens
        mean_n = 0.5*(self.dens + other.dens)
        y = dp / dn
        
        #y = y * (np.abs(y) < 3)

        #mask = (np.abs(dn) < 0.03)
        #y[mask] = np.nan  
        
        #mask = y >= threshold #-0.21
        #y[mask] = np.nan  
        
        result = PhiScan(y, self.rho, mean_n)
        result.dn = dn
        return result

    def __sub__(self, other): 
        dp = self.phi - other.phi
        dn = self.dens - other.dens
        result = PhiScan(dp, self.rho, dn)
        return result
        
    def plot(self):
        plt.plot(self.rho, self.phi)




def gen_pair_comb(L):
    for i in range(0, L):
        for j in range(i+1, L): 
            yield (i, j)

    

class PhiScanSet: 
    def __init__(self, scan_list=None): 
        self.scans = scan_list if scan_list is not None else []

    def plot(self): 
        for sc in self.scans: 
            sc.plot()
    
    @classmethod
    def load_scans(cls, shot, loader): 
        scans = PhiScan.load_scans(shot, loader)
        return cls(scans)

    def add_scans(self, shot, loader): 
        scans = PhiScan.load_scans(shot, loader)
        self.scans.extend(scans)

    def pairwise_normdiff(self): 
        L = len(self.scans)
        dscans = [self.scans[i] % self.scans[j] for i, j in gen_pair_comb(L)]
        result = PhiScanSet(dscans)
        return result

#def agglomerate(scans):
#    result = np.zeros((3, 0))
#    for sc in scans:
#        result = 
