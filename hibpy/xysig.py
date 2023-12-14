# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:45:17 2019

@author: reonid

Exports:  
    XYSignal

"""

from copy import deepcopy, copy as weakcopy
from inspect import getfullargspec 

#import numbers # Number, Complex, Real, Rational, Intergral
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import integrate as integ

from .cienptas import update_default_kwargs
from .winutils import copyPlainText, pastePlainText


class SignalError(Exception): 
    pass

#------------------------------------------------------------------------------
    
def search_sorted_ex(array, value): 
    L = len(array)
    i = np.searchsorted(array, value)
    if i <= 0: 
        return (0, 0, 1.0)
    elif i >= L: 
        return (L-1, L-1, 0.0)
    else: 
        v0 = array[i-1]
        v1 = array[i]
        t = (value-v0)/(v1-v0)
        return (i-1, i, t)

def value_at(x, xdata, ydata): 
    i0, i1, t = search_sorted_ex(xdata, x)
    y0, y1 = ydata[i0], ydata[i1]
    return y0 + (y1 - y0)*t

def as_array_ex(obj, copy=False): 
    if obj is None: 
        result = None
    else: 
        result = np.asarray(obj)
        if copy: 
            result = result.copy()
    return result

#------------------------------------------------------------------------------

def _get_func_fitparam_count(func): 
    if hasattr(func, '_fitparam_count_'): # used for functions with (*args)
        return func._fitparam_count_
    else: 
        argspec = getfullargspec(func)
        return len(argspec.args)-1

def _get_param(name, func, kwargs, default=None): 
    param = kwargs.get(name, None)
    
    if param is None: 
        if hasattr(func, '__dict__'):
            param = func.__dict__.get(name, default)
        else: 
            param = default

    return param


def curve_fit_ex(func, x, y, **kwargs): 
    '''
    Wrapper of scipy.optimize.curve_fit
    arguments of curve_fit can be stored as attributes of func
    '''
    
    nparam = _get_func_fitparam_count(func)
    
    p0             = _get_param('p0',             func, kwargs, None)
    sigma          = _get_param('sigma',          func, kwargs, None)
    absolute_sigma = _get_param('absolute_sigma', func, kwargs, False)
    check_finite   = _get_param('check_finite',   func, kwargs, True)
    bounds         = _get_param('bounds',         func, kwargs, (-np.inf, np.inf))
    method         = _get_param('method',         func, kwargs, None)
    jac            = _get_param('jac',            func, kwargs, None)
    maxfev         = _get_param('maxfev',         func, kwargs, 10**6)

    if p0 is None:     p0 = (1.0,)*nparam
    if bounds is None: bounds = [(-10,)*nparam, (10,)*nparam]

    defaults = {'p0': p0, 'sigma': sigma, 
                'absolute_sigma': absolute_sigma, 
                'check_finite': check_finite, 
                'method': method, 'jac': jac, 'maxfev': maxfev}
    
    curve_fit_args = getfullargspec(curve_fit)

    if 'bounds' in curve_fit_args.args: # .. versionadded:: 0.17
        defaults['bounds'] = bounds 
        
    kwargs = update_default_kwargs(curve_fit, defaults, kwargs)
    return curve_fit(func, x, y, **kwargs)
    
#------------------------------------------------------------------------------
    
class XYSignal: 
    # Class Flags
    allow_resample_in_binary_operators = False
    
    def __init__(self, xy=(None, None), copy=False):
        self.shot = None        
        self.name = ''
        self._setxy(xy, copy)
        
        self._fitopt = None
        self._fitfunc = None
        self._filename = None

    @classmethod
    def fromtxt(cls, filename, cols=(0, 1)): 
        result = cls()
        result.loadtxt(filename, cols)
        return result

    @classmethod
    def fromcsv(cls, filename, cols=(0, 1), **kwargs): 
        result = cls()
        result.loadcsv(filename, cols, **kwargs)
        return result
    
    @classmethod
    def fromdata(cls, xy): 
        return cls(xy)

    @classmethod
    def fromfunc(cls, f, x, *args, **kwargs): 
        y = f(x, *args, **kwargs)
        return cls((x, y))

    def __iter__(self):  
        '''
        Allows to unpack Signal object as
            x, y = sig
        '''
        yield self.x
        yield self.y

    def _setxy(self, xy, copy=False): 
        _x, _y = xy
        
        self.y = as_array_ex(_y, copy)
        self.x = as_array_ex(_x, copy)

    def __getitem__(self, arg):  
        '''
        Allows to make slice Signal
            fragment = sig[10:20]
        '''
        if isinstance(arg, slice): 
            return self.get_slice(arg)
        elif isinstance(arg, np.ndarray): 
            return self.get_obj_indexing(arg)
        elif isinstance(arg, int): 
            # to be consistent with unpack 
            if   arg == 0: return self.x
            elif arg == 1: return self.y
            else: raise SignalError("invalid index")
        else:
            # mask, etc ??? 
            return self.get_obj_indexing(arg)
            

    def get_slice(self, slc): 
        return self.get_obj_indexing(slc)

    def get_obj_indexing(self, obj_index):
        result = weakcopy(self)
        result._setxy((self.x[obj_index], self.y[obj_index]))
        return result

    def fragment(self, t0, t1=None): 
        if t1 is None: 
            t0, t1 = t0       # first arg can be tuple (t0, t1)
        j0 = np.searchsorted(self.x, t0)
        j1 = np.searchsorted(self.x, t1)
        return self[j0:j1+1]  # endpoint not included

    def zipxy(self): 
        return zip(self.x, self.y)
    
    def has_same_mesh_as(self, other): 
        if self.x.shape != other.x.shape: 
            return False
        else: 
            return np.isclose(self.x, other.x)
    
    def resample_as(self, refsig): 
        '''
        ??? Return new signal or modify self ???
        Now it modifies the original signal
        For copy use copy_resample_as
        '''

        refx = refsig.x if isinstance(refsig, XYSignal) else refsig
        new_x = deepcopy(refx)
        new_y = np.interp(new_x, self.x, self.y)
        self.x, self.y = new_x, new_y

    def copy_resample_as(self, refsig): 
        result = weakcopy(self)
        result.resample_as(refsig)
        return result

    def plot(self, *args, **kwargs):
        return plt.plot(self.x, self.y, *args, **kwargs)

    def savetxt(self, filename): 
        data = np.vstack((self.x, self.y))
        np.savetxt(filename, data.T, newline='\n')
        # self._filename = filename

    #def loadtxt(self, filename, cols=(0, 1), **kwargs): 
    #    data = np.loadtxt(filename, usecols=cols, **kwargs)
    #    self.x, self.y = data

    def loadtxt(self, filename, cols=(0, 1)): 
        data = np.loadtxt(filename)
        xcol, ycol = cols
        self.x = data[:, xcol]
        self.y = data[:, ycol]
        self._filename = filename

    def loadcsv(self, filename, cols=(0, 1), **kwargs): 
        data = np.loadtxt(filename, delimiter=',', 
                                usecols = cols, **kwargs)
        self.x, self.y = data[:, 0], data[:, 1]

    def transform(self, func): 
        result = weakcopy(self)
        result.y = func(self.y)
        return result
    
    def valueat(self, x): 
        return value_at(x, self.x, self.y)

    def sortx(self):
        order = self.x.argsort()
        self.x = self.x[order]
        self.y = self.y[order]

    def diff(self): 
        dy = np.diff(self.y)
        dx = np.diff(self.x)
        xx = 0.5*(self.x[0:-1] + self.x[1:])
        #result = deepcopy(self)
        
        #result = XYSignal((xx, dy))
        result = XYSignal((xx, dy/dx))
        
        result.name = 'diff ' + self.name
        result.resample_as(self)
        
        # correction of the first and the last points 
        #xx, yy = result.x, result.y
        #df_left  = (yy[ 2] - yy[ 1])/(xx[ 2] - xx[ 1])
        #df_right = (yy[-3] - yy[-2])/(xx[-3] - xx[-2])
        
        #result.y[0] = yy[1] - df_left*(xx[1] - xx[0])      # OK
        #result.y[-1] = yy[-2] - df_right*(xx[-2] - xx[-1]) # OK
        
        result.y[0 ] =  0.5/dx*( -3.0*self.y[0 ] + 4.0*self.y[1 ] - self.y[2 ] )
        result.y[-1] = -0.5/dx*( -3.0*self.y[-1] + 4.0*self.y[-2] - self.y[-3] )

        return result

    def append(self, x, y): 
        if self.x is None: 
            if isinstance(x, np.ndarray): # ??? isinstance(y, np.ndarray)
                self.x, self.y = x, y
            else:             
                self.x = np.array([x])
                self.y = np.array([y])
        else:
            self.x = np.append(self.x, x)
            self.y = np.append(self.y, y)
        
    def fit(self, func, mesh=None, **kwargs): 
        xx = np.linspace(-1.0, 1.0, 100) if mesh is None else mesh

        popt, pcov = curve_fit_ex(func, self.x, self.y, **kwargs)

        yy = func(xx, *popt)

        result = self.__class__() #XYSignal()
        result._setxy((xx, yy))
        result.name, result.shot = self.name, self.shot
        result._fitopt = popt
        result._fitfunc = func
        #self.smoothed = result
        return result


    def integrate(self, method): 
        return integ.trapz(self.y, x=self.x) # ??? method='simps'

    def normalize(self, kind=None): # on maximum, on 
        maxv = max(self.y)
        self.y = self.y/maxv
        
    def _get_other_operand(self, other):
        # return other.y if isinstance(other, XYSignal) else other
        if isinstance(other, XYSignal):
            if self.__class__.allow_resample_in_binary_operators: 
                if self.has_same_mesh_as(other): 
                    return other.y
                other_ = other.copy_resample_as(self)
                return other_.y
            else:     
                return other.y
        else: 
            return other
        
    def __neg__(self): 
        return XYSignal((self.x, -self.y))

    def __add__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, self.y + operand))

    def __sub__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, self.y - operand))

    def __mul__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, self.y * operand))

    def __truediv__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, self.y / operand))


    def __radd__(self, other): 
        return self + other

    def __rsub__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, operand - self.y))

    def __rmul__(self, other): 
        return self*other

    def __rtruediv__(self, other): 
        operand = self._get_other_operand(other)
        return XYSignal((self.x, operand/self.y))

    @classmethod
    def assemble_data(cls, signals, fmt='xyxy', resample=False): 
        data = None
        _ref = None
    
        for sig in signals: 
            if data is None: 
                _ref = sig
                if (fmt != "yyyy"): 
                    data = np.vstack((sig.x, sig.y))
                else:     
                    data = sig.y
            else: 
                if resample: 
                    _sig = sig.copy_resample_as(_ref)
                else: 
                    _sig = sig
    
                if fmt == 'xyxy': 
                    data = np.vstack((data, _sig.x))
                elif (fmt == 'xyyy') or (fmt == "yyyy"): 
                    pass
                else:
                    raise Exception('fmt should be "xyxy", "xyyy" or "yyyy" ')
    
                data = np.vstack((data, _sig.y))    
    
        return data.T

    def histogram(self, bins=None, **kwargs): 
        return HistogramSignal.build(self, bins, **kwargs)
        
    @classmethod    
    def paste_from_clipbrd(cls, xcol=0, ycol=1):
        plain_text = pastePlainText()
        
        list1 = plain_text.split('\r\n')
        list1 = list1[0:-1]
        list2 = [s.split('\t') for s in list1]

        L = len(list2)
        xx = np.zeros(L, dtype=np.float64)
        yy = np.zeros(L, dtype=np.float64)
        for i in range(0, L): 
            xx[i] = list2[i][xcol]
            yy[i] = list2[i][ycol]

        return XYSignal((xx, yy))

    def copy_to_clipbrd(self): 
        plain_text = '\r\n'.join([("%f\t%f" % (x, y) ) for x, y in zip(self.x, self.y) ])
        copyPlainText(plain_text)
        

    
#------------------------------------------------------------------------------    

class HistogramSignal(XYSignal): 
    def __init__(self, bin_edges, cnts): 
        L = len(cnts)
        xx = np.zeros(L)
        for i in range(L): 
            xx[i] = 0.5*( bin_edges[i] + bin_edges[i+1] )
        
        yy = cnts
        super().__init__((xx, yy))
        
        self._bin_edges = bin_edges

    @classmethod
    def build(cls, signal, bins=None, **kwargs): 
        ydata = signal.y if isinstance(signal, XYSignal) else signal
        
        if bins is None: 
            L = len(ydata)
            bins = int(2*L**0.35)+1

        cnts, bin_edges = np.histogram(ydata, bins, **kwargs)
        
        #result = HistogramSignal(bin_edges, cnts)
        result = cls(bin_edges, cnts)
        
        if isinstance(signal, XYSignal): 
          result.name = 'histogram ' + signal.name
          result.shot = signal.shot
        else: 
          result.name = 'histogram'
          result.shot = None
                  
        return result
        
    @property 
    def bins(self):       
        for a, b in zip(self._bin_edges, self._bin_edges[1:]): 
            yield (a, b)

    def get_slice(self, slc): 
        result = super.get_slice(slc)
        if (slc.step is None)or(slc.step == 1): 
            result._bin_edges = result._bin_edges[slc.start:slc.stop+1]
        else: 
            result._bin_edges = None
        return result 


if __name__ == '__main__':
    pass

