# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:25:10 2023

@author: reonid
"""


import numpy as np
#import matplotlib.mlab as mlab
#import scipy
import matplotlib.pyplot as plt

import dft

from .hibpcarp import psd, sampling_rate

SQRT_2PI = 2.506628274631000502415765284811

def normgauss(x, x0, sigma): # delta-function
  _x = x - x0;
  y = 0.5*(_x/sigma)**2
  return 1.0/(sigma*SQRT_2PI)*np.exp(-y)

#xxx = np.linspace(-100.0, 100.0, 1000)
#yyy = normgauss(xxx, 1.0, 2.0)
#plt.plot(xxx, yyy)
#plt.figure()
#print( 'simps =', scipy.integrate.simps(yyy, xxx) ) # !!!
#print( 'trapz =', scipy.integrate.trapz(yyy, xxx) )
#print( 'quad =', scipy.integrate.quad(normgauss, -100.0, 100.0, (1.0, 2.0) ) )
#print( 'quad =', scipy.integrate.quad(lambda x: normgauss(x, 1.0, 2.0), -10.0, 10.0 ) )

def gaussian(x, sigma):
    return np.exp(-x*x/(2.0*sigma**2) )


class SpectralGen: 
    def __init__(self, spectralfunc): # power spectrum  P(f)  # normalization:sigview (old matlab)
        self.spectralfunc = spectralfunc

    def generate_spectrum(self, times): 
        N = len(times)
        freqs = dft.freqs(times)
        freqs = dft.rearrange(freqs, '0+-', 'freq')
        powers = self.spectralfunc( abs(freqs) )
        amplitudes = powers**0.5
        
        phases = np.random.uniform(-np.pi, np.pi, N )
        phases[N//2+1:] = -phases[N//2-1:0:-1] # ???
        
        spectrum = amplitudes*(np.cos(phases) + 1.0j*np.sin(phases) )
        return spectrum*N**0.5 # / 0.15**0.5

    def generate(self, times, rms=None): 
        spectrum = self.generate_spectrum(times)
        data = np.fft.ifft(spectrum)
        
        if rms is not None: 
            _rms = np.std(data)
            data *= rms/_rms

        #return data 
        return np.real(data)

    def plot(self): 
        freqs = np.linspace(0, 1000.0, 500)
        powers = self.spectralfunc(freqs)
        
        plt.plot(freqs, powers)

if __name__ == "__main__": 
    
    N = 100000
    times = np.linspace(0.0, N*0.001, N)
    qcgen = SpectralGen(lambda x: gaussian(x - 150.0, 30.0) )
    yy = qcgen.generate(times)
    Pyy, ff = psd(yy, 1024, sampling_rate(times),  window=None, noverlap=256, sides='onesided')
    #Pyy, ff = sigv_psd(yy, 1024*2, sampling_rate(times),  window=None, noverlap=0, sides='onesided')
    #Pyy, ff = mlab.psd(yy, 1024, sampling_rate(times),  window=None, noverlap=256)
    plt.figure()
    plt.plot(ff, Pyy)
    
    qcgen.plot()
    
    data = qcgen.generate(times, rms = 0.1)
    rms = np.std(yy)
    print(rms)
    