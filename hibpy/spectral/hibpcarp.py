# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:21:47 2019

@author: reonid
Exports:  
    Classes: 
        Carpet
        CarpetRegion

    functions:     
        carpet(filename)
        carpet1(sig, func, nfft, winlen, ...)
        carpet2(sig1, sig2, func, nfft, winlen, ...)

"""

import numpy as np
from copy import deepcopy

from matplotlib import path
import matplotlib.mlab as mlab
#from numpy.fft import fft 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmath

from ..cienptas import update_default_kwargs
from ..loadsig import SignalsAreNotCompatible
from ..xysig import XYSignal
from ..hibpsig import signal #, Signal
from ..bcache import bread

#------------------------------------------------------------------------------
        
class Carpet: 
    def __init__(self, xyz = (None, None, None), copy=False):
        self.name = ''
        self.t, self.f, self.z = xyz
        if copy: 
            self.t = deepcopy(self.t)
            self.f = deepcopy(self.f)
            self.z = deepcopy(self.z)
        self.mask = None
        self.clim = None
        self.masklevel = 0
        self.fNyqvist = None
        self.from_sigview = False
        
    def __iter__(self):   # t, f, z = carp
        yield self.t
        yield self.f
        yield self.z
      
    # aliases:  x = t, y = f  
    def get_x(self): return self.t    
    def set_x(self, x): self.t = x    
    x = property(get_x, set_x)

    def get_y(self): return self.f    
    def set_y(self, y): self.f = y
    y = property(get_y, set_y)

    @property
    def shape(self): 
        if self.z is not None: 
            return self.z.shape
        else:
            return (0, 0)
    
    def masked_z(self): 
        if self.mask is None: 
            return self.z
        else: 
            if isinstance(self.mask, Carpet): 
                _mask = self.mask.z
            else: 
                _mask = self.mask
            #clean_func = np.vectorize(lambda z, m: z if m >= self.masklevel else np.nan)
            #return clean_func(self.z, _mask)
            _z = deepcopy(self.z)
            _z[_mask < self.masklevel] = np.nan
            return _z
    
    def xy(self):
        return (self.t, self.f)
        
    def loadfile(self, filename): 
        if filename.lower().endswith('.carp'):
            self._loadbin(filename)
        else: 
            self.loadtxt(filename)           

    def loadtxt(self, filename): 
        data = np.loadtxt(filename)
        data = data.T
        self.name = filename
        self.t = data[1:, 0]        
        self.f = data[0, 1:]
        self.z = data[1:, 1:]
        
        if (self.mask is not None) and (self.z.shape != self.mask.shape):
            self.mask = None
        
    def savetxt(self, filename): 
        data = np.zeros((len(self.t)+1, len(self.f)+1), dtype=self.z.dtype)
        data[1:, 0] = self.t           
        data[0, 1:] = self.f    
        data[1:, 1:] = self.z
        np.savetxt(filename, data.T, newline='\n')
    
    def _loadbin(self, filename):
        with open(filename, "rb") as file: 
            xL, el, x_bytes = bread(file, '_buf')
            yL, el, y_bytes = bread(file, '_buf')
            zL, el, z_bytes = bread(file, '_buf')
            mL, el, m_bytes = bread(file, '_buf')

            cmin = bread(file, 'd') 
            cmax = bread(file, 'd') 
            self.clim = (cmin, cmax)
            self.masklevel = bread(file, 'd')
            self.name = bread(file, '_str')
    
            #bread(file, 'i32')  # mask_color
            #bread(file, 'bool') # show_cmapl
            #bread(file, 'i32')  # orig_y_len
            file.read(9)    
    
            self.fNyqvist = bread(file, 'd')
           
            self.t = np.frombuffer(x_bytes, np.float64, xL)
            self.f = np.frombuffer(y_bytes, np.float64, yL)
            zz = np.frombuffer(z_bytes, np.float64, zL)
            zz = zz.reshape((yL, xL))
            self.z = zz.T     
     
            if m_bytes is None: 
                self.mask = None
            else:     
                mm = np.frombuffer(m_bytes, np.float64, mL)
                mm = mm.reshape((yL, xL))
                self.mask = mm.T

            if self.fNyqvist == 0.0:
                self.fNyqvist = self.f[-1]

            self.from_sigview = True 

    @classmethod
    def fromfile(cls, filename):
        carp = cls()
        carp.loadfile(filename)
        return carp    


    def yslice(self, j=None, y=None, f=None): 
        if j is None: 
            if y is None: 
                y = f
            j = np.searchsorted(self.y, y)
        return XYSignal((self.x, self.z[:, j]))
        
    def xslice(self, i=None, x=None, t=None): 
        if i is None: 
            if x is None: 
                x = t
            i = np.searchsorted(self.x, x)
        return XYSignal((self.y, self.z[i, :]))
        
    def plot(self, **kwargs):
        kwargs = update_default_kwargs(plt.imshow, kwargs, 
            {'cmap': cm.jet, 'aspect': 'auto', 'interpolation': 'bilinear', 'origin': 'lower'})
        
        zz = self.masked_z()        
        ax_im = plt.imshow(zz.T, 
                  extent = [self.t[0], self.t[-1], self.f[0], self.f[-1]], 
                  **kwargs)

        fig = ax_im.figure 
        self.scroll_event_id = fig.canvas.mpl_connect('scroll_event', onWhealEvent)

        if self.clim is not None: 
            ax_im.set_clim(self.clim)
            
    def calc_quantity(self, regionmask, quantity='ampl'): 
        if isinstance(regionmask, CarpetRegion): 
            regionmask = regionmask.calc_mask(carpet=self)
        
        #region_widths = np.sum(regionmask, axis=1)
        df = self.fNyqvist/len(self.f)
        
        z_in_rgn = deepcopy(self.masked_z())
        z_in_rgn[regionmask < 0.5] = np.nan
         
        if quantity == 'ampl': 
            integral = df * np.nansum(z_in_rgn, axis=1)
            ampl = np.sqrt(integral*2.0/self.fNyqvist)
            return XYSignal((self.x, ampl))
        if quantity == 'mean': 
            mean = np.nanmean(z_in_rgn, axis=1)
            return XYSignal((self.x, mean))

    def remove_chan(self, chan): 
        self.z = np.delete(self.z, chan, axis=0)
        self.z = np.delete(self.z, chan, axis=1)
        self.t = np.delete(self.t, chan)
        self.f = np.delete(self.f, chan)
        if self.mask is not None: 
            self.mask = np.delete(self.mask, chan, axis=0)
            self.mask = np.delete(self.mask, chan, axis=1)


def onWhealEvent(event):
    if event.name == 'scroll_event': 
        ax = plt.gca()
        cmin, cmax = ax.images[0].get_clim()

        if event.button == 'down':  k = 1.1
        elif event.button == 'up':  k = 0.9

        cmax *= k
        if not abs(cmin)*10 < abs(cmax): cmin *= k
        
        ax.images[0].set_clim((cmin, cmax))        
        plt.draw()


class CarpetRegion: 
    def __init__(self, x=None, y=None, name=None):
        self.name = name
        self.x = deepcopy(x)
        self.y = deepcopy(y)
        
        self.path = self._calc_path()
    
    def loadfile(self, filename):
        with open(filename, "rb") as file:  
            N = bread(file, 'i32')
            
            b = file.read(N*2*8)
            data = np.frombuffer(b, np.float64, N*2)
            self.x = data[::2 ] # odd
            self.y = data[1::2] # even
            
            bread(file, 'i32') # marker size
            bread(file, 'b') #        shape
            bread(file, 'b') #        style
            
            file.read(5) # visible for 5 slaves
            bread(file, 'i32') # color 
            self.name = bread(file, '_s') 
            #bread(file, '_propblock')
            
            self.path = self._calc_path()
    
    def _calc_path(self): 
        if (self.x is None) or (self.y is None):
            return None
        
        pts = [(x, y) for x, y in zip(self.x, self.y)]
        pts.append(pts[0])   # ???
        result = path.Path(pts, closed=True, readonly=True)
        return result
        
    @classmethod
    def fromfile(cls, filename):
        result = cls()
        result.loadfile(filename)
        return result        

    @classmethod
    def fromextent(cls, xmin, ymin, xmax, ymax, name=None):
        xx = np.array([xmin, xmin, xmax, xmax])
        yy = np.array([ymin, ymax, ymax, ymin])
        result = cls(xx, yy, name)
        return result
    
    def ptin(self, ptx, pty):
        return self.path.contains_point( (ptx, pty) )

    def plot(self):
        x = np.append(self.x, self.x[0])
        y = np.append(self.y, self.y[0])
        plt.plot(x, y)
        
    def calc_mask(self, xgrid=None, ygrid=None, carpet=None, dtype=np.float64): 
        if carpet is not None: 
            mesh_x, mesh_y = np.meshgrid(carpet.x, carpet.y)
        else: 
            mesh_x, mesh_y = np.meshgrid(xgrid, ygrid)

        mesh_pts = np.stack((mesh_x, mesh_y)).T   # 3D array
        result = np.zeros_like(mesh_x.T, dtype=dtype)
        
        for pt_slice, res_slice in zip(mesh_pts, result): 
            # pt_slice  array of points 
            res_slice[:] = self.path.contains_points( pt_slice)

            #pts = [(x, y) for x, y in pt_slice]
            #res_slice[:] = self.path.contains_points( pts )
            
        return result
        
        '''
        xx, yy = xarr, yarr
        M, N = len(xx), len(yy)
        result = np.zeros((M, N), dtype='bool')
        for i, x in enumerate(xx): 
            for j, y in enumerate(yy):
                x = xx[i]
                y = yy[j]
                if self.ptin(x, y):
                    result[i, j] = True
        return result
        '''     
 
#------------------------------------------------------------------------------

def sampling_rate(time_arr): 
    dx = (time_arr[-1] - time_arr[0])/len(time_arr)
    return 1.0/dx


def _carpet2(signal1, signal2, func, zone_len, NFFT, win_len, window=None, 
             stepdiv=2, **kwargs): 
    '''
    There is some imcompatibility with arguments in MATLAB signature and 
    python mlab signature: 
    python NFFT corresponds to win_length
    python pad_to corresponds to NFFT
    '''
    
    if isinstance(func, str): 
        func = mlab.__dict__[func]
    
    xx, yy1 = signal1
    
    if signal2 is not None: 
        xx2, yy2 = signal2 
        if not np.allclose(xx, xx2, 1e-6): 
            raise SignalsAreNotCompatible('Signals have different sampling')   
        
        
    L = len(xx)
    rate = sampling_rate(xx)

    kwargs = update_default_kwargs(func, kwargs, 
        {'detrend': 'linear', 'scale_by_freq': True, 'noverlap': win_len//2})
        
    if window is None: 
        window = mlab.window_hanning
            
    start = 0
    result = None
    time = []
    while True: 
        if start + zone_len > L: break
        chunk1 = yy1[start:start + zone_len]
        chunk1 = mlab.detrend_linear(chunk1)
        mean_x = 0.5*( xx[start] + xx[start + zone_len - 1] )
        
        if signal2 is not None: 
            chunk2 = yy2[start:start + zone_len]
            chunk2 = mlab.detrend_linear(chunk2)
            y, freq = func(chunk1, chunk2, Fs=rate, NFFT=win_len, pad_to=NFFT, 
                           window=window, **kwargs) 
        else: 
            y, freq = func(chunk1,         Fs=rate, NFFT=win_len, pad_to=NFFT, 
                           window=window, **kwargs) 
                                        
        if result is None: 
            result = y
        else:              
            result = np.vstack((result, y))  
        
        time.append(mean_x)
        start += zone_len // stepdiv
    
    time = np.array(time)
    
    carp = Carpet((time, freq, result))
    carp.fNyqvist = rate * 0.5
    carp.name = func.__name__
    return carp

#------------------------------------------------------------------------------

def cross_phase(x, y, NFFT=None, Fs=None, detrend='linear', window=None, noverlap=None, **kwargs):
    
    '''
    ??? not compatible yet ???
    '''
    Pxy, freqs = mlab.csd(x, y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, 
                          noverlap=noverlap, scale_by_freq=True, **kwargs)
    return np.angle(Pxy), freqs

#def psd(x, NFFT, Fs, **kwargs):
def psd(x, NFFT=None, Fs=None, detrend='linear', window=None, noverlap=None, **kwargs):
    '''
    psd with the same normalization as in the SigViewer in case sides='onesided' (or None for Real input array)
    '''
    Pxx, freqs = mlab.psd(x, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, 
                          noverlap=noverlap, scale_by_freq=True, **kwargs)
    fNyqvist = 0.5*Fs

    return Pxx*fNyqvist, freqs

def csd(x, y, NFFT=None, Fs=None, detrend='linear', window=None, noverlap=None, **kwargs):
    '''
    csd with the same normalization as in the SigViewer
    '''
    Pxy, freqs = mlab.csd(x, y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, 
                          noverlap=noverlap, scale_by_freq=True, **kwargs)
    fNyqvist = 0.5*Fs
    return Pxy*fNyqvist, freqs

def re_csd(x, y, NFFT=None, Fs=None, detrend='linear', window=None, noverlap=None, **kwargs):
    '''
    re_csd with the same normalization as in the SigViewer
    '''

    Pxy, freqs = csd(x, y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, 
                          noverlap=noverlap, scale_by_freq=True, **kwargs)
    return np.real(Pxy), freqs

class SigviewSpectralFunctions: 
    '''
    Namespace for spectral functions 
    that are compatible with SigView functions
    '''
    def __init__(self): 
        self.psd = psd
        self.csd = csd
        self.phase = cross_phase
        self.cohere = mlab.cohere
        self.re_csd = re_csd

sigv = SigviewSpectralFunctions()

#def frac(x): 
#    return  x % 1

def confine_phase(a): # -pi .. +pi
    x = a/2.0/np.pi + 0.5
    x = x % 1 # frac(x)
    x = (x - 0.5)*2.0*np.pi
    return x

vect_confine_phase = np.vectorize(confine_phase)

#------------------------------------------------------------------------------

def carpet1(signal1, func, zone_len, win_len, window=None, **kwargs): 
    NFFT = zone_len    
    return _carpet2(signal1, None, 
                    func, zone_len, NFFT, win_len, window, **kwargs)
    

def carpet2(signal1, signal2, func, zone_len, win_len, window=None, **kwargs):     
    NFFT = zone_len    
    return _carpet2(signal1, signal2, 
                    func, zone_len, NFFT, win_len, window, **kwargs)

def carpet(filename): 
    carp = Carpet()
    carp.loadfile(filename)
    return carp

#------------------------------------------------------------------------------

if __name__ == '__main__':

    sig1 = signal(44381, "ABOL4", device='tjii')
    sig2 = signal(44381, "ABOL5")
    s1 = sig1.fragment(1000, 1300)    
    s2 = sig2.fragment(1000, 1300)    

    carph = carpet2(s1, s2, mlab.csd, 1024*2, 256*2)
    carph.z = np.angle(carph.z)

    carph.clim = (-3, 3)
    #carph.mask = carp.z
    carph.mask = carpet2(s1, s2, mlab.cohere, 1024*2, 256*2)
    carph.masklevel = 0.3    

    carph.plot()

