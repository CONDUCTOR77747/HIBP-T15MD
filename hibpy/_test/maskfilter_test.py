# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:54:43 2020

@author: reonid
"""
import numpy as np
import matplotlib.pyplot as plt

from maskfilter import mftr, _inverse_cmp_op

TIME = mftr('time')
SIG = mftr('sig')


time = np.linspace(0, 20.0, 200)
sig = np.sin(time)

#plt.plot(time, sig)

mask = np.ones_like(time, dtype='bool')

ftr = (SIG > 0.8) + (SIG < -0.8)

ftr.clean(mask, name_provider = globals())

filtered_sig = sig[mask]
_min = np.min(filtered_sig)
_max = np.max(filtered_sig)

assert _min > -0.8 
assert _max < 0.8 

ftr = SIG.inside(-0.9, 0.9) # inside(SIG, 0.3, 0.6)
ftr += SIG > 0.95

mask = np.ones_like(time, dtype='bool')
ftr.clean(mask, name_provider = globals())

filtered_sig = sig[mask]
filtered_t = time[mask]


plt.plot(filtered_t, filtered_sig)


for op in _inverse_cmp_op.keys(): 
    iop = _inverse_cmp_op[op]
    iiop = _inverse_cmp_op[iop]
    if iiop != op: 
        raise Exception('%s != %s' % (iiop, op))

