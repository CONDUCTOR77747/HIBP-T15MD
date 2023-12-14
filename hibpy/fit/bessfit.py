# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:18:36 2021

@author: reonid

"""

import numpy as np
import scipy.special as spf

j0roots = spf.jn_zeros(0, 1000)

def _basis_func(n):
    nthroot = j0roots[n]
    def func(x): 
        return spf.j0(x*nthroot)
    
    return func

basis_funcs = [_basis_func(i) for i in range(0, 20)]
    
def _fitf(x, *args): 
    result = np.zeros_like(x)
    for i, a in enumerate(args):    
        result = result + a*basis_funcs[i](x)
    return result

def bessfit1(x, p0): 
    return p0*basis_funcs[0](x)

def bessfit2(x, p0, p1): 
    return p0*basis_funcs[0](x) + p1*basis_funcs[1](x)

def bessfit3(x, p0, p1, p2): return _fitf(x, p0, p1, p2)
def bessfit4(x, p0, p1, p2, p3): return _fitf(x, p0, p1, p2, p3)
def bessfit5(x, p0, p1, p2, p3, p4): return _fitf(x, p0, p1, p2, p3, p4)
def bessfit6(x, p0, p1, p2, p3, p4, p5): return _fitf(x, p0, p1, p2, p3, p4, p5)
def bessfit7(x, p0, p1, p2, p3, p4, p5, p6): return _fitf(x, p0, p1, p2, p3, p4, p5, p6)


#def _polyf(x, pw, *args): 
#    pass

def polyfunc(pw): 
    '''
    Polynomial parametric function with arbitrary powers
    pw: list of powers, for example
      polyfunc([0, 2, 4]) for f(x) = p0 + p1*x^2 + p2*x^4
    
    '''
    
    def f0(x, p0):                             return p0*x**pw[0]
    def f1(x, p0, p1):                         return p0*x**pw[0] + p1*x**pw[1]
    def f2(x, p0, p1, p2):                     return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2]
    def f3(x, p0, p1, p2, p3):                 return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3]
    def f4(x, p0, p1, p2, p3, p4):             return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4]
    def f5(x, p0, p1, p2, p3, p4, p5):         return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5]
    def f6(x, p0, p1, p2, p3, p4, p5, p6):     return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5] + p6*x**pw[6]
    def f7(x, p0, p1, p2, p3, p4, p5, p6, p7): return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5] + p6*x**pw[6] + p7*x**pw[7]
    
    def f8(x, p0, p1, p2, p3, p4, p5, p6, p7, p8): 
        return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5] + p6*x**pw[6] + p7*x**pw[7] + p8*x**pw[8]

    def f9(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9): 
        return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5] + p6*x**pw[6] + p7*x**pw[7] + p8*x**pw[8] + p9*x**pw[9]
    
    def f10(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10): 
        return p0*x**pw[0] + p1*x**pw[1] + p2*x**pw[2] + p3*x**pw[3] + p4*x**pw[4] + p5*x**pw[5] + p6*x**pw[6] + p7*x**pw[7] + p8*x**pw[8] + p9*x**pw[9] + p10*x**pw[10]

    N = len(pw)
    if N > 11: 
        raise Exception("Too many powers")

    return locals()['f%d' % (N-1) ]


