# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:53:10 2023

@author: reonid

Lomb-Scargle periodogram

X=X[j], t=t[j], Σ=Σ
                  ʲ
                  
tg(2ωτ) = Σsinω(t-τ) / Σcosω(t-τ)             
             
P(ω) = 0.5·{ [ΣX·cosω(t-τ)]²/Σcos²ω(t-τ) + [ΣX·sinω(t-τ)]²/Σsin²ω(t-τ) }
P(ω) = 0.5·{ A²(ω) + B²(ω) }

A(ω) = [ΣX·cosω(t-τ)]/Σcosω(t-τ)
B(ω) = [ΣX·sinω(t-τ)]/Σsinω(t-τ)

X(t) = ΣA(ω)cosωt + ΣB(ω)sinωt

"""

import numpy as np
#import matplotlib.mlab as mlab   #         y = mlab.detrend_linear(y)
import matplotlib.pyplot as plt
#import numpy.fft as fft
import math

from astropy.stats import LombScargle

import fourier
#import numbers
#import math
#import cmath
#from numpy import sin, cos


PI = np.pi
_2PI = 2.0*np.pi

#%%

def calc_tau(omega, tt): 
    #if omega == 0.0: 
    #    return 0.0 # ??
    ss = np.sin(2.0*omega*tt)
    cc = np.cos(2.0*omega*tt)
    theta = math.atan( np.sum(ss)/np.sum(cc) )  # θ = 2ωτ
    #theta = math.atan2( np.sum(ss), np.sum(cc) )  # ??? θ = 2ωτ
    return theta/(2.0*omega)

def cos2divider(omega, tt, tau): # sqrt( Σcos²ω(t-τ) )
    arg = omega*(tt - tau)
    cc = np.cos(arg)
    cc2 = cc**2
    return np.sum(cc2)**0.5

def sin2divider(omega, tt, tau): # sqrt( Σsin²ω(t-τ) )
    arg = omega*(tt - tau)
    ss = np.cos(arg)
    ss2 = ss**2
    return np.sum(ss2)**0.5

# Cosinusoidal coefficiant A(ω) 
def Aw(omega, tt, yy, tau=None): 
    if tau is None:  
        tau = calc_tau(omega, tt)

    arg = omega*(tt - tau)
    cc = np.cos(arg)
    yycc = yy*cc
    result = np.sum(yycc)
    result = result / cos2divider(omega, tt, tau)
    return result

# Sinusoidal coefficiant A(ω) 
def Bw(omega, tt, yy, tau=None):
    if tau is None:  
        tau = calc_tau(omega, tt)

    arg = omega*(tt - tau)
    ss = np.sin(arg)
    yyss = yy*ss
    result = np.sum(yyss)
    result = result / sin2divider(omega, tt, tau)
    return result

# Power P(ω) 
def Pw(omega, tt, yy, tau=None):
    if tau is None:  
        tau = calc_tau(omega, tt)

    A = Aw(omega, tt, yy, tau)
    B = Bw(omega, tt, yy, tau)
    return 0.5*(A**2 + B**2)

# Power P(ω), in alternative form. Just for test
def Pw_(omega, tt, yy, tau=None):
    if tau is None:  
        tau = calc_tau(omega, tt)

    arg = omega*(tt - tau)
    cc = np.cos(arg)
    ss = np.sin(arg)

    yycc = yy*cc
    yyss = yy*ss

    return 0.5*(  np.sum(yycc)**2 / np.sum(cc**2) + 
                  np.sum(yyss)**2 / np.sum(ss**2)  )


def _recovered(ww, aa, bb, tt): # ???
    result = np.zeros_like(tt)
    for omega, a, b in zip(ww, aa, bb): 
        if np.isnan(a): continue
        if np.isnan(b): continue
    
        cc = np.cos(omega*tt)
        ss = np.sin(omega*tt)
        result = result + a*cc + b*ss
    return result

class MyLombScargle: 
    def __init__(self, tt, yy, **kwargs): 
        self.tt = tt
        self.yy = yy
        
    def complexcoeffs(self, freqs):
        omegas = freqs*_2PI
        aa = np.array( [Aw(w, self.tt, self.yy) for w in omegas] )
        bb = np.array( [Bw(w, self.tt, self.yy) for w in omegas] )        
        result = aa - 1.0j*bb
        return result
    
    def power(self, freqs): 
        cc = self.complexcoeffs(freqs)
        return np.abs(cc)**2

if __name__ == '__main__': 
    
    tag = 0
    
    if tag == 0: 
        N = 24
        probe_numbers = np.linspace(0, N-1, N)
        thetas = probe_numbers * _2PI/N
        
        # Merezhkin transform
        merezhkin_lambda = 0.7
        thetas_ = thetas - merezhkin_lambda*np.sin(thetas)#
        
        poloidal_m = 3
        yy = np.sin(thetas_*poloidal_m)
        plt.figure()
        plt.plot(thetas, yy)
        
        Fyy = np.fft.fft(yy)
        Pyy = abs(Fyy)**2
        mm = fourier.freqs(thetas)*_2PI     # ~ omegas 
        plt.figure()
        plt.plot(mm, Pyy, 'o-')
    
        #mm_ = np.linspace(mm[0], mm[N//2], N)
        mm_ = np.linspace(1.0, 12.0, 12)
        Pyy_ = np.array( [Pw(m, thetas_, yy)*N      for m in mm_] )
        Ayy_ = np.array( [Aw(m, thetas_, yy)*N**0.5 for m in mm_] )
        Byy_ = np.array( [Bw(m, thetas_, yy)*N**0.5 for m in mm_] )
        
        #plt.figure()
        plt.plot(mm_, Pyy_, 'o-')    
    
        #yy_ = recovered(mm[0:N//2], Ayy_, Byy_, thetas_)/N*2
        yy_ = _recovered(mm_, Ayy_, Byy_, thetas_)/N 
        plt.figure()
        plt.plot(thetas, yy)
        

        plt.plot(thetas, yy_)
        
        ls = MyLombScargle(thetas_, yy)
        #ff, pp = ls.autopower()
        
        plt.figure()
        pp = ls.power(mm/_2PI)
        plt.plot(mm, pp)

        ls = MyLombScargle(thetas_, yy)
        plt.figure()
        #plt.plot(ff*_2PI, pp)
        
        pp = ls.power(mm/_2PI)
        plt.plot(mm, pp)

    elif tag == 1: 
        tt  = [0.1,  0.2,  0.3,  0.4,  0.5,   0.6,  0.7,  0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5]   
        _tt = [0.1,  0.12, 0.15, 0.2,  0.25,  0.3,  0.4,  0.5,  0.6,  0.8,  1.0,  1.2,  1.3,  1.4,  1.5]
    
        tt = np.array(tt)
        yy = np.sin(tt*15) #*TWO_PI) # + 0.4*np.sin(tt*2.3*TWO_PI)
        
        max_dt = np.max(tt[1:] - tt[:-1]) 
        min_dt = np.min(tt[1:] - tt[:-1]) 
        max_w = 1.0/max_dt*_2PI 
        #max_w = 1.0/min_dt*_2PI
     
        ww = np.linspace(-50.90, 60.0, 1000)  # omegas 
        pp = np.array( [Pw(w, tt, yy) for w in ww] )
        
        plt.figure()
        plt.plot(ww, pp)
        
        plt.figure()
        plt.plot(tt, yy)
        
        aa = np.array( [Aw(w, tt, yy) for w in ww] )
        bb = np.array( [Bw(w, tt, yy) for w in ww] )
        
        #plt.figure()
        ww_ = ww
        tt_ = tt
        
        #ww_ = np.linspace(-max_w/2, max_w/2, 1000)
        #tt_ = np.linspace(0.0, 2.0, 100)
        
        yy_ = _recovered(ww_, aa, bb, tt_)
        plt.plot(tt_, yy_*0.00273)
    


