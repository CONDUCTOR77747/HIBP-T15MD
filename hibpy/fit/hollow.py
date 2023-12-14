# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:36:14 2019

@author: reonid

Exports:  
    hollow_func(x, alpha, ampl) 
    norm_hollow_func(x, alpha) 
    van_milligen_func(x, p1, p2, p3)
    adapt4fit(f, scaling=False)
    fit(x, y, func, mesh=None)    
    
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from inspect import getfullargspec 
from scipy.optimize import curve_fit 

#import time
#from contextlib import contextmanager

def hollow_func(x, ampl, alpha): 
    # ampr = 1.335033 --> f(0) = 1
    return ampl*norm_hollow_func(x, alpha)
    
def norm_hollow_func(x, alpha): 
    """
     Uni-parametric family of functions 
               hollow --> parabolic
     alpha =     0.07 --> 1
     x = -1..+1
    
     Intergal from -1 to 1 gives 1.0 (with an error less than 1%)
                                    less than 0.3% if alpha >= 0.1
     Function supports vector call (for fit)
    """
    alpha = abs(alpha)
    mean_approximation = 0.5045 - 0.0097*math.pow(alpha, -0.5)
    corectional_norm = 0.5/mean_approximation

    result = _f4(x, alpha, 1.0)
    return result*corectional_norm


def _f1(x, alpha, beta): 
    y = (x+1.0)*beta
    return y/(y*y + alpha)


def _f2(x, alpha, beta): 
    return _f1(x, alpha, beta) + _f1(-x, alpha, beta)


def _f3(x, alpha, beta=1.0): 
    shift = 2.0*beta/(alpha+4.0*beta*beta)
    result = _f2(x, alpha, beta) - shift 

    norm = math.log(alpha + 4.0*beta*beta) - math.log(alpha)
    norm = norm/beta
    norm = norm - 2.0*shift

    return result/norm


def _f4(x, alpha, beta=1.0):
   if alpha < 0.05: alpha = 0.05

   gamma = 1 - 0.05/alpha*abs(x)
   #x = np.sign(x)*math.pow(abs(x), gamma)
   x = np.sign(x)*np.power(abs(x), gamma)
   return _f3(x, alpha, beta)

#------------------------------------------------------------------------------

def van_milligen_func(x, p1, p2, p3): 
    '''
    B. van Milligen approximation function
    '''
    x_2 = x*x
    x_4 = x_2*x_2
    result = ( p1*(1.0 - x_2) + p2*(1.0 - x_4) )*np.exp(-p3*x_2)
    return result

def van_milligen_func_c(x, p1, p2, p3, c): 
    x_2 = x*x
    x_4 = x_2*x_2
    result = ( p1*(1.0 - x_2) + p2*(1.0 - x_4) )*np.exp(-p3*x_2)
    return result + c

#------------------------------------------------------------------------------

def hokusai_func2(x, alpha): 
    beta = 1.0
    result = beta/(alpha+x*x)
    f_1 = beta/(alpha+1.0)
    f_0 = beta/alpha - beta/(alpha+1.0)
    return (result - f_1)/f_0


def _f12(x, gamma): 
    gamma = gamma**2
    result = np.exp(-np.abs(x)*gamma)
    result = result*(1.0 + gamma*np.abs(x))
    #result = (1.0 + gamma*np.abs(x)) 
    return result
    
def norm_hokusai_func(x, gamma): 
    result = _f12(x, gamma) 
    f0 = _f12(0.0, gamma)
    f1 = _f12(1.0, gamma)
    return (result - f1)/(f0 - f1)

def hokusai_func(x, ampl, gamma): 
    return ampl*norm_hokusai_func(x, gamma)

#------------------------------------------------------------------------------

def test_vect_support(f, skiptest=False): 
    '''
    checks the possibility to call with np.array as first argument 
    for parametric functions like f(x, param1, param2, ...) 

    can return wrong result if value 0.5 is out of domain for some parameters
    '''
    if skiptest: 
        return False
    
    #test_arr = np.array([], dtype=np.float64)
    test_arr = np.array([0.02, 0.98], dtype=np.float64)
    fa = getfullargspec(f) 
    args = [0.5]*(len(fa.args)-1) 
    try: 
        f(test_arr, *args) # unused = f(test_arr)
    except: 
        return False
    else: 
        return True

def _get_func_fitparam_count(f): 
    if hasattr(f, '_fitparam_count_'): 
        return f._fitparam_count_
    else: 
        argspec = getfullargspec(f)
        return len(argspec.args)-1


def adapt4fit(f, scaling=False, skiptest=False): 
    '''
    Adapts function of scalar argument (and arbitrary parameters)
    for using in scipy.optimize.curve_fit 
    Resulting function can be called both in vector form and in scalar form
    
    If scaling == True then additional scaling parameter added to original parameter set
    (It is useful for normalized functions) 
    
    '''
    
    vf = np.vectorize(f)    
    nparam = _get_func_fitparam_count(f)
    
    def vectf(x, *args): 
        if isinstance(x, np.ndarray): 
            #return np.array([f(_x, *args) for _x in x])
            return vf(x, *args)
        else: 
            return f(x, *args)
    vectf._fitparam_count_ = nparam
    
    def vectf_with_scaling(x, ampl, *args): 
        if isinstance(x, np.ndarray): 
            #return ampl*np.array([f(_x, *args) for _x in x])
            return ampl*vf(x, *args)
        else: 
            return ampl*f(x, *args)
    vectf_with_scaling._fitparam_count_ = nparam + 1
    
    def f_with_scaling(x, ampl, *args): 
        return ampl*f(x, *args)
    f_with_scaling._fitparam_count_ = nparam + 1

    if scaling: 
        result_f = f_with_scaling if test_vect_support(f, skiptest) else vectf_with_scaling
    else:       
        result_f = f if test_vect_support(f, skiptest) else vectf

    return result_f

def fit(x, y, func, mesh=None, p0=None, maxvef=None, bounds=None): 
    nparam = _get_func_fitparam_count(func)
    
    if p0 is None: 
        p0 = (1.0,)*nparam
    
    if bounds is None: 
        bounds = [(-10,)*nparam, (10,)*nparam]

    curve_fit_args = getfullargspec(curve_fit)
    if 'bounds' in curve_fit_args.args: # .. versionadded:: 0.17
        popt, pcov = curve_fit(func, x, y, p0, bounds=bounds, maxfev=10**6) 
    else:  # old version of scipy   (Anaconda for WinXP)
        popt, pcov = curve_fit(func, x, y, p0, maxfev=10**6) 
    
    def f(x): 
        return func(x, *popt)
    
    if mesh is None: 
        return f, popt
    else: 
        return func(mesh, *popt), popt

#------------------------------------------------------------------------------
   
def __test(): 
    xarray = np.linspace(-1.0, 1.0, 1000)

    alphas = [0.5, 0.8, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.99]
    #alphas = np.linspace(0.1, 1.0, 10)

    for i in range(len(alphas)): 
        yarray = [hollow_func(x, 1.0, alphas[i]) for x in xarray]
        plt.plot(xarray, yarray)


#  test integral     
def __test2(): 
    xarray = np.linspace(-1.0, 1.0, 1000)

    #alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.99]

    alphas = np.linspace(0.1, 1.0, 100)

    means = np.empty(len(alphas))

    for i in range(len(alphas)): 
        yarray = [hollow_func(x, 1.0, alphas[i]) for x in xarray]
        means[i] = np.mean(yarray)
        #plt.plot(xarray, yarray)

    plt.plot(alphas, means) 

#------------------------------------------------------------------------------

if __name__ == '__main__':
    __test2()
    f = adapt4fit(hollow_func)
    print(f is hollow_func)

