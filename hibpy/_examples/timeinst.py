# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:01:26 2019

@author: Kurchatov
"""

from hibpsig import signal
from xysig import idx2x
from cienptas import find_ranges, narrow_mask 
#import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
#from copy import deepcopy



def steady_mask(sig, eps, dead_zone=None): 
    L = len(sig.y)
    h = sig.histogram()
    
    if dead_zone is not None: 
        h.y[h.x < dead_zone] = 0
 
    i_max = np.argmax(h.y)
    nominal = h.x[i_max]
    
    if (h.y[i_max] < L // 10): 
        mask = np.zeros(L)
    else: 
        mask = (nominal-eps <= sig.y) & (sig.y <= nominal+eps)
    
    return mask

shots = []
#sh = open('E:\\Lists\\regimes\\2.3-230.txt')
with open('J:\\TestShared\\2019\\hibplib\\2.3-230.txt', 'r') as sh:
    for line in sh:
        nn = line.split('!')[0].split('-')
        n0 = int(nn[0])
        n1 = int(nn[1])
        rng = range(n0, n1+1)
        shots.extend(rng)
        
# shot = 62753

fname = 'J:\\TestShared\\2019\\hibplib\\time_inst.txt'
lim = [1.8, 2.2]
times = []    

for shot in shots:
    
    Ipl = signal(shot, "I.I{avg333, x-1}", device='T10')
    Ipl = Ipl.fragment(345, 1000)
    
    Ipl.y = savgol_filter(Ipl.y, 51, 3)
    
    mask_Ipl = steady_mask(Ipl, 2, 90)
    mask_Ipl = narrow_mask(mask_Ipl, left=500, right=50)
    
    if np.all(mask_Ipl == 0): 
        continue
    
    ne = signal(shot, "I.f8{x0.1333, z200, avg33}", device='T10')
    EC = signal(shot, "I.EC{x-1, z80, avg21n5}", device='T10')

    irng_steady_Ipl = find_ranges(mask_Ipl, lambda m: m)[0]
    xrng_steady_Ipl = idx2x(irng_steady_Ipl, Ipl.x)
    if xrng_steady_Ipl[1] - xrng_steady_Ipl[0] < 12: 
        continue

    Ipl = Ipl.fragment(xrng_steady_Ipl)
    ne  = ne.fragment (xrng_steady_Ipl)
    EC  = EC.fragment (xrng_steady_Ipl)

    mask_EC = (EC.y < 0.02)
    mask_EC = narrow_mask(mask_EC, left=10, right=100)
    mask_ne = (ne.y < lim[1]) & (ne.y > lim[0])
    mask_ne = narrow_mask(mask_ne, left=5, right=5)
    mask = mask_EC & mask_ne

    idx_rngs = find_ranges(mask, lambda x: x == 1)            # indices 
    time_rngs = idx2x(idx_rngs, ne.x)    
        
    #t = [(ne.x[i0], ne.x[i1-1]) for i0, i1 in rngs]  # times
    #t = [(int(t0), int(t1)) for (t0, t1) in t if t1 - t0 > 1] 

    if len(time_rngs) != 0:
        for tt in time_rngs: 
            if tt[1] - tt[0] < 5: 
                continue
            times.append([shot, tt[0], tt[1]])
            ne.fragment(  tt[0], tt[1]  ).plot()
            #Ipl.fragment( tt[0], tt[1]  ).plot()
    print(shot)

np.savetxt(fname, times, fmt = '%5d') 

        
        
        

        
        
'''
        mask = (Ipl.y < Ipl_max + 5)&(Ipl.y > Ipl_max - 5) # & (EC_ampl < 0.02) & (ne.y < 2.2) & (ne.y > 1.8)
        t = Ipl.x[mask]

        
 #       EC_ampl = np.array(EC.y)
 #       ne_ampl = np.array(ne.y)

        
        if t.size != 0:
                        
            print(shot)
            
            for x in range(len(t)):
                shot_time = [shot, t[x]]
#           shot_time = [shot, min(t), max(t)]
                times.append(shot_time)
#                print(times)
            
np.savetxt(fname, times, fmt = '%5d')     
        
# plt.plot(ne.x, ne.y)
# plt.plot(Ipl.x, Ipl.y)

#    I_div = np.gradient(I_pl, Ipl.x)
#    I_div = savgol_filter(I_div, 51, 3)

#plt.plot(Ipl.x, I_div)

    

#print(min(t), max(t))
 '''       
