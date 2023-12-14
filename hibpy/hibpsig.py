# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:45:17 2019

@author: reonid

Exports:  
    Signal
    HistogramSignal
    ThomsonSignal    
    
    signal(shot, name, source='sigloader')
                                    'cache'
                                    'file'
                                    'TS'
                                    
    thomson(shot, name, calib=False)

    e_beam(shot)
    eboard_param(shot, dtype=int)
    eboard_details(shot, asdict=True)

    shot_list(txt)
    sig_list(txt)
    
    config_loader(device='tjii')
"""

import numpy as np

from .loadsig import loadSignal, sendLoaderCmd, loadInfo
from .bcache import loadCacheFile, loadCachedSignal, loadThomsonSignal, parseCacheFileName
from .xysig import XYSignal


class Signal(XYSignal): 
    def __init__(self, xy=(None, None), copy=False):
        super().__init__(xy, copy)
        
    # aliase:  t = x
    def get_t(self): return self.x    
    def set_t(self, t): self.x = t    
    t = property(get_t, set_t)

    def load(self, shot, name, **kwargs): 
        self.t, self.y, signame = loadSignal(shot, name, **kwargs)
        self.shot = shot
        self.name = signame #name
        
    def loadfile(self, filename): 
        if filename.lower().endswith('.cache'): 
            self.t, self.y = loadCacheFile(filename)
            _, self.shot, self.name = parseCacheFileName(filename)
        else: 
            self.loadtxt(filename)
    
    def loadcache(self, shot, name, **kwargs):
        self.t, self.y = loadCachedSignal(shot, name)
        self.shot = shot
        self.name = name

#------------------------------------------------------------------------------

def signal(shot, name, source='sigloader', **kwargs): 
    sig = Signal()
    if source == 'sigloader': 
        sig.load(shot, name, **kwargs)
    elif source == 'cache': 
        sig.loadcache(shot, name)
    elif source == 'file': 
        sig.loadtxt(name, **kwargs)
    elif (source == 'TS')or(source == 'thomson'): 
        sig.shot = shot
        sig.name = name
        (sig.x, sig.y), t = loadThomsonSignal(shot, name)
    return sig

def spectral_quantity_in_rgn(shot, signal1, signal2=None, func='psd', 
                             nfft=1024, winlen=512, region=None, quantity='ampl', 
                             masklevel=None, stepdiv=None, **kwargs): 
    '''
    Examples: 
    sig = spectral_feature_in_rgn(50261, 'HIBPII::Itot{slit3}', func='psd', nfft=1024, winlen=512, 
                                  region='d:\\2d\\test.rgn', quantity='ampl', dtype='float64')    
    
    sig2 = spectral_feature_in_rgn(50261, 'HIBPII::Itot{slit3}', func='psd', nfft=1024, winlen=512, 
                                  region='d:\\2d\\test.rgn', quantity='rms', dtype='float64')    
    
    sig = spectral_feature_in_rgn(50261, 'HIBPII::Itot{slit2}', 'HIBPII::Itot{slit4}', func='phase', nfft=1024*2, winlen=512, 
                              region='d:\\2d\\test2.rgn', quantity='mean', masklevel=0.7, dtype='float64')     
    '''
    
    paramstr = (f'&SPECTRAL shot={shot} sig1="{signal1}" sig2="{signal2}" '
                f'specfunc={func} nfft={nfft} winlen={winlen} region="{region}" '
                f'rgnfunc={quantity} masklev={masklevel} stepdiv={stepdiv}')
    return signal(shot, paramstr, 'sigloader', **kwargs)


#------------------------------------------------------------------------------
    
def e_beam(shot, hibp=2): 
    if hibp in [0, 1, 2]: 
        return sendLoaderCmd(shot=int(shot), hibp_param='Ebeam', hibp=hibp)

def eboard_param(shot, param_name, dtype=float):
    txt = loadInfo(shot, '?DETAILS::' + param_name)
    if (dtype is str)or(dtype is None): 
        return txt
    else:
        return dtype(txt)

def eboard_details(shot, asdict=True):
    txt = loadInfo(shot, '?DETAILS::All')
    strlist = [s.strip() for s in txt.split('\r\n')]
    strlist = [s for s in strlist if s != '']
    if asdict:
        paramdict = {}
        for s in strlist: 
            k, v, *_  = s.split(':')
            k, v = k.strip(), v.strip()
            paramdict[k] = v
        return paramdict
    else: 
        return strlist
    
def shot_list(s): 
    txt = loadInfo(shot=0, cmd='?SHOTLIST::' + s)
    return [int(s) for s in txt.split('\r\n') if s !='']

def sig_list(s): 
    txt = loadInfo(shot=0, cmd='?SIGLIST::' + s)
    return [s for s in txt.split('\r\n') if s !='']

def config_loader(**kwargs): 
    sendLoaderCmd(None, **kwargs)

#------------------------------------------------------------------------------
    
if __name__ == '__main__':
    pass
 