# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:27:32 2023

Methodical utilites for 

        Direct Fourier Transform

@author: reonid

Notation of representation of the data and frequency

0++       Standard representation. f[0] = 0, f[N//2] = fNyq. Frequencies at the second half are artificial (f_actual = f_art - 2.0*fNyq)
0+-       Standard order of spectrum, but frequencies at the second half are negative. !!! Frequencies are not monotonic  
-0+       Spectra ordered by signed frequency. f[N//2-1] = 0, f[-1] = fNyq
0+        Half of spectra (length = N//2+1). Negative frequencies discarded. !!! Necessary to scale spectra to keep power the same (except f=0 and f=fNyq) !!! 


From cos/sin form to exp form: 
    
   C = 0.5*(A - i·B)  # for positive frequencies
   C(-ω) = C*(+ω)

    

N                      N   
Σ(A·cosωt + B·sinωt) = Σ C·exp(-iωt)
0                     -N+1 



"""


import numpy as np
import matplotlib.pyplot as plt
#import math

PI = np.pi
_2PI = 2.0*np.pi
_2PI_I = 2.0j*np.pi

#%%

def direct_fourier(data): 
    N = len(data)
    result = np.zeros_like(data, dtype=np.complex)
    for k in range(N): 
        for j in range(N): 
            _exp = np.exp(-_2PI_I*k*j/N)
            result[k] += _exp*data[j]
            
    #result /= N     
    return result

def inverse_fourier(Fdata): 
    N = len(Fdata)
    result = np.zeros_like(Fdata, dtype=np.complex)
    for k in range(N): 
        for j in range(N): 
            _exp = np.exp(_2PI_I*k/N *j)
            result[k] += _exp*Fdata[j]
            
    result /= N
    return result  

def inverse_fourier_continuous(x, original_xarray, Fdata): # zero-based
#def inverse_fourier_continuous(x, xx_Fdata): # zero-based
#    xx, Fdata = xx_Fdata
    
    N = len(Fdata)
    i_Nyq = N // 2
    xx = original_xarray
    _x = (x - xx[0])*(N-1)/(xx[-1] - xx[0])
    
    result = 0.0j
    for k in range(N): 
        
        w = _2PI*k/N
        if k > i_Nyq:     # ! SIC  artificial freq -> actual negative freq
            w = - _2PI*(N - k)/N
            
        _exp = np.exp(1.0j*w*_x)
        result += Fdata[k]*_exp
    return result/N


#%%

def freqs(times): # in '0++' representation
    # !!! assert( times is regular )
    L = len(times)  # NFFT
    dt = times[1] - times[0]
    ff = np.arange(L)/dt/L
    return ff

def omegas(times): 
    return freqs(times)*_2PI

#%%

def _fourier_rearrange(data): # "0++" -> "-0+"
    N = len(ff)
    N_ = N//2 + 1
    _data = np.empty_like(data)
    
    _data[N_-2: ] = data[0:N_]
    _data[0:N_-2] = data[N_: ]
    return _data

def fourier_freq_representation(ff, view): # representation, view, form, order, rearrangement 
    # !!! assert( N is even )
    N = len(ff)
    N_ = N//2 + 1
    fNyq = ff[N_-1]
    _ff = ff.copy() 
    
    if (view == 'std')or(view == 'raw')or(view == '0++'): 
        pass

    elif (view == 'half')or(view == '0+'): 
        _ff = ff[0:N_].copy()

    elif (view == '0..+f..-f')or(view == 'signed')or(view == '0+-'): 
        _ff[0:N_] = ff[0:N_]               # positive
        _ff[N_: ] = ff[N_: ] - 2.0*fNyq    # negative
    
    elif (view == '-f..+f')or(view == 'centered')or(view == '-0+'): 
        _ff[N_-2: ] = ff[0:N_]             # positive
        _ff[0:N_-2] = ff[N_: ] - 2.0*fNyq  # negative
        
    return _ff

def fourier_data_representation(data, view):  # rescale='pwr', 'ampl', 'none'
    # !!! assert( N is even )
    N = len(data)
    N_ = N//2 + 1
    _data = data.copy()
    
    if (view == 'std')or(view == 'raw')or(view == '0++'): 
        pass

    elif (view == 'half')or(view == '0+'): 
        _data = data[0:N_]

    elif (view == '0..+f..-f')or(view == 'signed')or(view == '0+-'): 
        pass
    
    elif (view == '-f..+f')or(view == 'centered')or(view == '-0+'): 
        _data[N_-2: ] = data[0:N_] 
        _data[0:N_-2] = data[N_: ]

#    if (view == '0+')and(rescale == 'ampl'): 
#        _data[1:-1] *= 2.0
        
    return _data


def fourier_data_representation2d(data, view):
    _data = data
    _data = fourier_data_representation(_data, view)
    _data = fourier_data_representation(_data.T, view).T
    return _data

def rearrange(arr, view, kind):  # rescale='pwr', 'ampl', 'none' 
    '''
    
    '''
    
    kind = kind.replace(' ', '')
    if kind == 'freq': 
        return fourier_freq_representation(arr, view)
    elif kind == 'data': 
        return fourier_data_representation(arr, view)
    elif (kind == 'data2d')or(kind == '2d'): 
        return fourier_data_representation2d(arr, view)

    elif (kind == 'freq,data'): 
        freq, data = arr
        return ( fourier_freq_representation(freq, view), 
                 fourier_data_representation(data, view)   )

    elif (kind == 'time,freq,data2d'): 
        time, freq, data = arr
        return ( fourier_freq_representation(time, view), 
                 fourier_freq_representation(freq, view), 
                 fourier_data_representation2d(data, view)   )
    else: 
        raise ValueError('Invalid kind: %' % kind)
    
        
    
#_ff_ = np.array([0, 1, 2, 3, 4, 5, 6, 7])
#_ff_s = fourier_freq_representation(_ff_, 'signed')
#_ff_c = fourier_freq_representation(_ff_, 'centered')
#_ff_h = fourier_freq_representation(_ff_, 'half')
#print(_ff_)
#print(_ff_s)
#print(_ff_c)
#print(_ff_h)
#test = fourier_data_representation(_ff_, 'centered')
#test = fourier_data_representation(test, 'centered')


def _reduplicate(arr, n, delta=0.0): 
    result = arr
    for i in range(n-1): 
        result = np.hstack(  (result, arr + delta*(i+1) )  )
    return result    





if __name__ == '__main__': 
    xx = np.linspace(0.0, 1.0, 100) + 2.3
    yy = np.sin(xx*10*2*np.pi)

    #plt.plot(xx, yy)
    
    Fyy_ = np.fft.fft(yy)
    Fyy = direct_fourier(yy)


    ff = freqs(xx)
    plt.plot(ff, abs(Fyy))
    #plt.plot(ff,abs(Fyy_))
    
    _xx = np.linspace(0.0, 1.0, 1000) + 2.3
    
    plt.figure()
    #_yy = np.array([inverse_fourier_continuous(_x, xx, Fyy) for _x in _xx])
    _yy = inverse_fourier_continuous(_xx, xx, Fyy)
         
    plt.plot(_xx, np.real(_yy) )
    plt.plot(xx, np.real(yy) )


