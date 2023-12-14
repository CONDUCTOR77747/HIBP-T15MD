# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:34:40 2023

@author: reonid
"""


import numpy as np
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



#------------------------------------------------------------------------------

def _pade2func(a, b, c, d, e): 
    def f(x): 
        xx = x*x
        return (a + b*x + c*xx)/(1.0 + d*x + e*xx)
    
    def df(x): # df/dx
        xx = x*x
        A = (b - a*d) + 2.0*x*(c - a*e) + xx*(c*d - b*e)
        B = (1.0 + d*x + e*xx)**2
        return A/B
    
    return f, df


def pade2func0(a, d, e): # f'(0) = 0  # 
    b = a*d
    c = -(a + b)
    return _pade2func(a, b, c, d, e)

def pade2func0p(ampl, d, peaking): 
    a = ampl
    p = peaking    
    b = a*d
    c = -(a + b)
    e = p/a*(4.0*a + 2.0*b + c) - 4.0 - 2.0*d
    return _pade2func(a, b, c, d, e)



#------------------------------------------------------------------------------

def _pade3func(a, b, c, d, e, f, g): 
    def func(x): 
        xx = x*x
        xxx = x*xx
        return (a + b*x + c*xx + d*xxx)/(1.0 + e*x + f*xx + g*xxx)
    
    def dfunc(x): 
        xx = x*x
        xxx = x*xx
        xxxx = xx*xx
        a0 = b - a*e
        a1 = 2.0*(c - a*f)
        a2 = c*e + 3.0*(d - a*g) - b*f
        a3 = 2.0*(d*e - b*g)
        a4 = d*f - c*g
        A = a0 + a1*x + a2*xx + a3*xxx + a4*xxxx 
        B = (1.0 + e*x + f*xx + g*xxx)**2
        return A/B
    
    return func, dfunc

def pade3func0(a, c, e, f, g): 
    b = a*e           # f'(0) = 0
    d = - a - b - c   # f(1) = 0
    return _pade3func(a, b, c, d, e, f, g)
    
def pade3func0p(ampl, c, e, f, peaking): 
    a = ampl
    p = peaking
    
    b = a*e  # f'(0) = 0
    d = - a - b - c  # f(1) = 0
    g = p/a*(8.0*a + 4.0*b + 2.0*c + d) - 8.0 - 4.0*e - 2.0*f   # peaking

    return _pade3func(a, b, c, d, e, f, g)

#------------------------------------------------------------------------------

if __name__ == "__main__": 
    xx = np.linspace(-1.0*0.0, 1.0, 500)



    paramdiap = [1e-6, 1e-5, 0.0001,  0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 100.0, 300.0, 1000.0, 1e4, 1e5, 1e6]
    p = 1.1
    
    cnt = 0
    for c in paramdiap: 
        for e in paramdiap: 
            for f in paramdiap: 
                f, df = pade3func0p(1.0, c, e, f, p)
                yy = f(xx)
                if np.min(yy) < 0.0: continue
                #if f(0.0) < f(0.003): continue
                if f(0.2) > 1.2: continue
                if df(1.0) < -10.0: continue
                if f(0.03) < 0.998: continue
            
                cnt += 1 
                plt.plot(xx, yy)
    
    
    
    print('cnt = ', cnt)
    