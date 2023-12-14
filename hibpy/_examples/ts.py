# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:22:50 2020

@author: reonid
"""

import sys
hibplib_path = 'G:\\myPy\\reonid-packages'
if hibplib_path not in sys.path: sys.path.append(hibplib_path)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#from scipy import integrate

from hibplib.bcache import loadThomsonSignal

from hibplib.xysig import XYSignal
from hibplib.hollow import hollow_func, van_milligen_func, adapt4fit

from hibplib.thoms import ThomsonSignal, thomson_navg

shots = [33277, 33278,  33282, 33283, 33296]
#shots = [33277]
#shots = [38236]

shot = 33199   # 1160  0.36  1.0738
shot = 33266  # 1105  0.47  1.0791
shot = 38236  # 1130  0.97  1.6456

ne_avg = {33199:0.36, 33266:0.47, 38236:0.97, 49873:0.48, 49870:0.49, 49863:1.04}
thomson_navg.extend(ne_avg)

'''
ne = thomson(shot, 'Ne')

#ne = ne.fragment(-0.6, 0.7)

ne_ = ne.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])
#ne_ = ne.fit(adapt4fit(hollow_func, scaling=True), p0=(1, 1), maxfev=10**6, bounds=[(0.001, 0.07), (10, 1.0)])
#ne_ = ne.fit(hollow_func, p0=(1,), maxfev=10**6, bounds=[(0.07,), (1.0,)])

k = ne_avg[shot]/ne_.lineavg
ne.y *= k
ne_.y *= k

ne.plot()
ne_.plot()
#print(ne_.integral)

ne.savetxt('c:\\ne_raw.dat')
ne_.savetxt('c:\\ne_fit.dat')
'''

'''
Te = thomson(shot, 'Te')
#Te = Te.fragment(-0.6, 0.72)
#Te_ = Te.fit(adapt4fit(hollow_func, scaling=True), p0=(1, 1), maxfev=10**6, bounds=[(0.001, 0.07), (10, 1.0)])
Te_ = Te.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])
Te.plot()
Te_.plot()

Te.savetxt('c:\\Te_raw.dat')
Te_.savetxt('c:\\Te_fit.dat')
'''

shots_18_dec = [49856, 49854, 49879, 49857, 49858, 49861, 49863, 49864, 49867, 
                49868, 49869, 49870, 49871, 49872, 49873, 49874, 49875, 49876, 
                49880, 49878]

ts_high = [49856, 49863, 49864, 49867, 49868, 49875, 49876, 49878]
ts_low = [49854, 49858, 49861, 49869, 49870, 49873]



tag = 4.1

if tag == 0: 

    phi = ThomsonSignal()
    phi.loadtxt('j:\\reonid\\2019\\axel\\raw\\33266_phi.dat', (0, 1))
    phi = phi.fragment(-0.7, 0.9)
    #phi.y += 0.02
    
    plt.figure()
    phi.plot()
    phi_ = phi.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])
    phi_.plot()
    phi_.savetxt('c:\\Phi_fit.dat')

elif tag == 1: 
    Te = thomson(49863, 'Te')
    Te_ = Te.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])
    plt.figure()
    Te_.plot()
    Te.plot()

elif tag == 1.5: 
    #shot = 49839
    shot = 49866
    #shot = 33710
    ne = thomson(shot, 'Ne')
    #ne_ = ne.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])
    ne_ = ne.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6)

    #k = ne_avg[shot]/ne_.lineavg
    k = ne_avg.get(shot, 1)/ne_.lineavg
    #print(ne_avg[shot])
    print(k)
    #ne.y *= k
    #ne_.y *= k

    plt.figure()
    ne_.plot()
    ne.plot()
    
    
elif tag == 2: 
    plt.figure()
    for sh in shots_18_dec: 
        try: 
            Te = thomson(sh, 'Ne')
        except: 
            continue
        
        if Te.time > 1158: 
        #if Te.time < 1180: 
            continue
        
        Te_ = Te.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])

        Te_.plot()
        Te.plot()
        print('%d   %d' % (sh, Te.time))

elif tag == 3: 
    plt.figure()
    ne = thomson(ts_high, 'Ne')
    ne_ = ne.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])

    ne_.plot()
    ne.plot()
    print('%d   %d' % (sh, ne.time))

#Te = signal(49873, 'Te', source='TS')
#Te.plot()

elif tag == 4:
    shots = [33277, 33278,  33282, 33283, 33296]   
    
    ne = thomson(shots, 'Ne', calib=True)
    #ne_ = ne.fit(milligen_func, p0=(1, 1, 1), maxfev=10**6) #, bounds=[(-10, -10, -10), (10, 10, 10)])
    ne_ = ne.fit(adapt4fit(hollow_func, scaling=True), p0=(1, 1), maxfev=10**6) #, bounds=[(-10, -10, -10), (10, 10, 10)])

    ne_.plot()
    ne.plot()

elif tag == 4.1:
    shots = [33277, 33278,  33282, 33283, 33296]   
    
    Te = thomson(shots, 'Te', calib=True)
    Te = Te.fragment(-0.5, 0.67)    
    Te.plot()    
    Te.savetxt('d://neExp.txt')
    
elif tag == 10:
    data = None
    signame = "Te"
    for sh in shots:
        sig = loadThomsonSignal(sh, signame)
        if data is None: 
            data = sig
        else: 
            data = np.hstack((data, sig))
    
    #data = np.sort(data, axis=0, key=lambda a: a[0])
    permutation = data[0, :].argsort()
    data = data[:, permutation]
    
    #data = data[:, data[0, :] <  0.7]
    #data = data[:, data[0, :] > -0.4]
    
    
    
    condition = lambda x: (x > -0.5)and(x < 0.7)
    #condition = lambda x: (x > -0.9)and(x < 0.9)
    #condition = adapt4fit(condition, restype=bool)
    #condition = adapt4arr(condition)
    condition = np.vectorize(condition)
    data = data[:, condition(data[0, :]) ]
    
    
          
    fit_f = adapt4fit(hollow_func, scaling=True)
    popt, pcov = curve_fit(fit_f, data[0, :], data[1, :], (1, 1), maxfev=10**6, bounds=[(0.001, 0.07), (10, 1.0)])    
    
    
    xx = np.linspace(-1, 1, 100)
    #plt.plot(xx,  fit_f(xx, *popt))
    plt.plot(data[0, :], data[1, :])
    
    
    fit_f = hollow_func
    popt, pcov = curve_fit(fit_f, data[0, :], data[1, :], (1,), maxfev=10**6, bounds=[(0.07,), (1.0,)])    
    #plt.plot(xx,  fit_f(xx, *popt))
    
    
    fit_f = van_milligen_func
    popt, pcov = curve_fit(fit_f, data[0, :], data[1, :], (1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10), (10, 10, 10)])    
    plt.plot(xx,  fit_f(xx, *popt))
    
    #fit_f = BvM_func2
    #popt, pcov = curve_fit(fit_f, data[0, :], data[1, :], (1, 1, 1, 1), maxfev=10**6, bounds=[(-10, -10, -10, -10), (10, 10, 10, 10)])    
    #plt.plot(xx,  fit_f(xx, *popt))
    
    ts = XYSignal((xx,  fit_f(xx, *popt)))
    ts.savetxt("c:\%d_%s.dat" % (shots[0], signame))
    
    
    fit_y = fit_f(xx, *popt)
    fit_xy = np.vstack((xx, fit_y))
    #np.savetxt('d:\\test.txt', fit_xy.T)
    
    
    #big_arr = rand.uniform(-10, 10, (10000000))
    testf = lambda x: 1.2*x**3 + 2.0*x**2 + x + 10.0





#Ip_b4_{avg55n11, from1000to1140, rar10}
#Ip_b4_{avg55n11, from1066to1205, rar10}


# np.testing.assert_almost_equal(np.average(a.reshape(48, -1)

# np.array([np.average(a[i], weights=b) for i in range(48)]))
