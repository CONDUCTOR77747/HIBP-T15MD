# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:15:10 2022

@author: reonid
"""

import sys
hibplib_path = 'd:\\reonid\\myPython\\reonid-packages'
if hibplib_path not in sys.path: sys.path.append(hibplib_path)

#from copy import deepcopy
import numpy as np
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#from scipy import integrate



import matplotlib.mlab as mlab
#from matplotlib import path
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from matplotlib import colors as mcolors

import numpy.fft as npfft

#result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]

import .hibpcarp as hibpcarp

def boxcar(x): 
    return 1.0

sigv_psd = hibpcarp.psd
mlab_psd = mlab.psd
fft = npfft.fft


def gaussian(x, sigma):
    return np.exp(-x*x/(2.0*sigma**2) )

def signalgen(N, central_freq, width, Npeaks=40, df=0.1): 
    yy = np.zeros(N)
    for i in range(Npeaks): 
        f = central_freq + i*df
        yy += np.sin(xx*f + i)*gaussian(i - Npeaks//2, width)
    return yy

#%% test signal 
N = 1024*4
N_ = int(N/2) +1
xx = np.linspace(0.0, 100.0, N)
#yy = np.sin(xx*10.0) + 2.0*np.sin(xx*10.15) + np.sin(xx*10.3) + 0.5*np.sin(xx*10.45) + 0.25*np.sin(xx*10.6)

yy = signalgen(N, central_freq=40.0, width=8.0, Npeaks=40, df=0.1)


#plt.plot(xx, yy)

#%%  psd.m
'''
index = 1:nwind;
KMU = k*norm(window)^2;	% Normalizing scale factor ==> asymptotically unbiased
% KMU = k*sum(window)^2;% alt. Nrmlzng scale factor ==> peaks are about right

Spec = zeros(nfft,1); Spec2 = zeros(nfft,1);
for i=1:k
    if strcmp(dflag,'none')
        xw = window.*(x(index));
    elseif strcmp(dflag,'linear')
        xw = window.*detrend(x(index));
    else
        xw = window.*detrend(x(index),'constant');
    end
    index = index + (nwind - noverlap);
    Xx = abs(fft(xw,nfft)).^2;
    Spec = Spec + Xx;
    Spec2 = Spec2 + abs(Xx).^2;
end

% Select first half
if ~any(any(imag(x)~=0)),   % if x is not complex
    if rem(nfft,2),    % nfft odd
        select = (1:(nfft+1)/2)';
    else
        select = (1:nfft/2+1)';
    end
    Spec = Spec(select);
    Spec2 = Spec2(select);
%    Spec = 4*Spec(select);     % double the signal content - essentially
% folding over the negative frequencies onto the positive and adding.
%    Spec2 = 16*Spec2(select);
else
    select = (1:nfft)';
end
freq_vector = (select - 1)*Fs/nfft;

% find confidence interval if needed
if (nargout == 3)|((nargout == 0)&~isempty(p)),
    if isempty(p),
        p = .95;    % default
    end
    % Confidence interval from Kay, p. 76, eqn 4.16:
    % (first column is lower edge of conf int., 2nd col is upper edge)
    confid = Spec*chi2conf(p,k)/KMU;

    if noverlap > 0
        disp('Warning: confidence intervals inaccurate for NOVERLAP > 0.')
    end
end

Spec = Spec*(1/KMU);   % normalize

Pxx = Spec;

'''
#%%

'''
    # old matlab
    # KMU = k*norm(window)^2   
    #    where  k - number of windows
    
    
    if scale_by_freq: 
        # py-mat norm
        
        result /= Fs
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2.
        result /= (np.abs(windowVals)**2).sum()
    else:             
        # py-altern norm
        
        # In this case, preserve power in the segment, not amplitude
        result /= np.abs(windowVals).sum()**2
'''

#%%

def norm(xx): 
    return ( np.sum(xx)/len(xx) )**2

def norm2(xx): 
    return np.sum(xx**2)/len(xx)


#%%

#winf = mlab.window_none 
winf = mlab.window_hanning

Fs = hibpcarp.sampling_rate(xx)
carp_Pyy0, ff = sigv_psd(yy, NFFT=N, Fs=Fs, detrend='linear', window=winf, noverlap=None)                       # Sigview norm - old Matlab norm
mlab_Pyy1, ff = mlab_psd(yy, NFFT=N, Fs=Fs, detrend='linear', window=winf, noverlap=None, scale_by_freq=False)  # alternative Python norm 
mlab_Pyy2, ff = mlab_psd(yy, NFFT=N, Fs=Fs, detrend='linear', window=winf, noverlap=None, scale_by_freq=True)   # default Python norm, scale_by_freq=True for MATLAB compatibility - new Matlab norm

Fyy = fft( winf(yy) )
k_win2 = 1.0/norm2( winf(np.ones_like(yy)) )
k_win  = 1.0/norm( winf(np.ones_like(yy)) )
fft_Pyy = np.abs(Fyy)**2
fft_Pyy = fft_Pyy[0:N_]


# how to translate all the norms to my one: 
plt.plot(ff,       fft_Pyy   * k_win2/N,             color="green")  
plt.plot(ff+0.001, carp_Pyy0,                        color="black")  # my 
plt.plot(ff+0.002, mlab_Pyy1 * N/2 * (k_win2/k_win), color="red")    # py-alt
plt.plot(ff+0.003, mlab_Pyy2 * Fs/2,                 color="blue")   # py-mat

# how to translate fft to other: 

plt.figure()
plt.plot(ff,       fft_Pyy*2*k_win/N/N,             color="green")  
plt.plot(ff+0.002, mlab_Pyy1,                       color="red")    # py-alt

plt.figure()
plt.plot(ff,       fft_Pyy*2*k_win2/N/Fs,           color="green")  
plt.plot(ff+0.002, mlab_Pyy2,                       color="blue")    # py-alt


'''
window_hanning
window_none
numpy.blackman
numpy.hamming
numpy.bartlett
scipy.signal.get_window

'''


