# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:16:40 2020

@author: reonid
"""

import numpy as np
import matplotlib.pyplot as plt
#from math import cosh
from numpy import cosh
from scipy import diff


from .hollow import adapt4fit, fit, van_milligen_func
from ..xysig import XYSignal
from ..hibpsig import signal


def triangle(x): 
    return (1.0 - abs(x))/1.0

def triangle1(x, h): 
    '''
    rounded triangle: edges are sharp
    '''
    if h > 0.5: h = 0.5

    x = abs(x)

    if x > 1.0: 
        return 0.0
    elif x > h: 
        return 1.0 - x
    else: # xx <= h
        return 1.0 - (x*x + h*h)*0.5/h;


def triangle2(x, h): 
    '''
    rounded triangle: edges are smooth
    '''
    if h > 0.5: h = 0.5

    x = abs(x)

    if x > 1.0 + h: 
        return 0.0
    elif x > 1.0 - h: 
        return (0.5 - x + h + np.square(x - h)*0.5)*0.5/h
    elif x > h: 
        return 1.0 - x
    else: # x <= h
        return 1.0 - (x*x + h*h)*0.5/h;

#------------------------------------------------------------------------------

def trapezium(x, d): 
    if d > 1.0: d = 1.0

    x = abs(x)
    if x >= 1.0: 
        return 0.0
    elif x <= d: 
        return 1.0
    else: 
        return triangle(x)/triangle(d)

def trapezium1(x, d, h): 
    '''
    rounded trapezium: edges are sharp
    '''
    if d > 1.0: d = 1.0
    if h > d: h = d

    half_slope = (d + 1.0)*0.5
    x = abs(x)

    if x > 1.0 + h: 
        return 0.0
    elif x < d - h: 
        return 1.0
    elif x < half_slope: 
        return 1.0 - triangle2((x-1.0)/(1.0-d), h)
    else: #if x > half_slope
        return triangle1((x-d)/(1.0-d), h)
    
def trapezium2(x, d, h): 
    '''
    rounded trapezium: edges are smooth
    '''
    if d > 1.0: d = 1.0
    if h > d: h = d
    
    half_slope = (d + 1.0)*0.5
    x = abs(x)
    
    if x > 1.0 + h: 
        return 0.0
    elif x < d - h: 
        return 1.0
    elif x < half_slope: 
        return 1.0 - triangle2((x-1.0)/(1.0-d), h)
    else: #if x > half_slope
        return triangle2((x-d)/(1.0-d), h)


def koppa_family(x, d, h, k): 
    return trapezium1(x, d, h) + k*triangle2(x/d, h)
    
def realistic_koppa(x): 
    '''
    Only shape, not coefficient !!!
    '''
    return koppa_family(x, 0.45, 0.3, 0.07)
    

def mexican_hat(x, k1, k2, d, h): 
    #return k1*triangle1(x, h) + k2*trapezium1(x, d, h)
    #return k1*triangle1(x, h) + k2*koppa_family(x, d, h, 0.3)
    return k1*triangle1(x, h) + k2*realistic_koppa(x)

def basedens_potential(shot, n0, hibpsignames): 
    phiname = hibpsignames['phi']
    densname = hibpsignames['dens']
    rhoname = hibpsignames['rho']
        
    phi = signal(shot, phiname)
    dens = signal(shot, densname)
    rho = signal(shot, rhoname)
    dens.resample_as(phi)
    rho.resample_as(phi)
    
    #corr = 1.15*adapt4fit(realistic_koppa)(rho.y)
    #corr = 1.0*adapt4fit(realistic_koppa)(rho.y)
    corr = 1.2*adapt4fit(realistic_koppa)(rho.y)
    corr = corr*(dens.y - n0)
    phi.y = phi.y  + corr
    return phi, rho, dens
    

if __name__ == '__main__': 
    
    mesh = np.linspace(-1.2, 1.2, 200)
    y = adapt4fit(trapezium)(mesh, 0.2)
    
    
    tag = 4
    
    if tag == 0: 
        for h in [0.05, 0.1, 0.2, 0.3, 0.4]: 
            y = adapt4fit(trapezium2)(mesh, 0.5, h)
            plt.plot(mesh, y)
    
    
    elif tag == 1: 
        for k2 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]: 
            y = adapt4fit(mexican_hat)(mesh, 1.0, -k2, 0.5, 0.1)
            #y = diff(y)
            #mesh = mesh[0:-1]
            plt.plot(mesh, y)

    elif tag == 2: 
        
        for k in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]: 
            y = adapt4fit(koppa_family)(mesh, 0.5, 0.2, k)
            plt.plot(mesh, y)
    
    elif tag == 3: 
        koppa_exp = XYSignal.fromtxt('M:\\Work\\2020\\pscaling\\koppa.txt')    
        koppa_exp.plot()
        
        y = adapt4fit(koppa_family)(mesh, 0.45, 0.3, 0.07)
        plt.plot(mesh, -1.15*y)

    elif tag == 4: 
        hibpsignames = {
          'phi': 'HIBPII::Phi{slit4, zdcorr0.09, avg91n11, rar20, %s}',            
          'rho': 'HIBPII::Rho{?M:\\rho\\2D_12mar\\privE%EII%_slit3.dat:1:2, rar200, dtime=-0.06}', 
          'dens': 'DENCM0_{avg11n11}'
          }
        shot = 50277
        phi, rho, ne = magic_potential(shot, 0.55, hibpsignames)
        #phi.plot() 
        
        phi0 = signal(shot, hibpsignames['phi'])
        #phi0.plot()
        
        phi = phi.fragment(1062.98, 1118.71)
        rho = rho.fragment(1062.98, 1118.71)
        plt.plot(rho.y, phi.y)



