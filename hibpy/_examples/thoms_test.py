# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:39:28 2021

@author: reonid
"""

import sys
hibplib_path = 'D:\\myPython\\reonid-packages'
if hibplib_path not in sys.path: sys.path.append(hibplib_path)

#import numpy as np
import matplotlib.pyplot as plt

from hibplib.fit.hollow import hokusai_func
from hibplib.fit.bessfit import bessfit2, bessfit3
from hibplib.thoms import thomson, thomson_navg
from hibplib.hibpsig import config_loader #  , shot_list, eboard_param 

thomson_fit_func_for_calib = bessfit2

ne_avg = {33199:0.36, 33266:0.47, 38236:0.97, 49873:0.48, 49870:0.49, 49863:1.04}
thomson_navg.update(ne_avg)



Ne = thomson(48428, 'Ne', calib=True)
#Ne = Ne.fragment(-1.0, 0.3)
Ne_ = Ne.fragment(-0.4, 0.8).fit(bessfit3)

plt.figure()
Ne.plot()
Ne_.plot()

gradNe_ = Ne_.diff()
gradNe_.y = gradNe_.y*5
gradNe_.plot()

Te = thomson([48427, 48428, 48424], 'Te') 
Te_ = Te.fragment(-1.0, 0.3).fit(hokusai_func)
Te.plot()
Te_.plot()