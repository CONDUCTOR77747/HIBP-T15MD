# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:42:25 2020

@author: reonid
"""

import os


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
#from matplotlib import cm
#from scipy import interpolate
#from scipy.signal import savgol_filter
#from scipy.optimize import curve_fit


from ..cienptas import func_from_file
from ..xysig import XYSignal
from ..hibpsig import signal #, Signal
#from magsur import MagConf

#%%

    
def add_option(signame, opt):
    if opt is None: 
        return signame
    elif not opt.strip(): # empty string
        return signame    
    else:
        return signame + '{' + opt + '}'
    '''
    elif signame.find('{') == -1: 
        return signame + '{' + opt + '}'
    else: 
        return signame.replace('}', ', ' + opt + '}')
    '''

class RadRef: 
    def __init__(self, path): 
        self.path = path

    def coord_func(Ebeam, coord): 
        #fname = 'd:\\rho\\2D\\mar2020_B096\\radref_E%d.dat' % Ebeam
        return lambda uscan: 0.0
    
    def xfunc(self, Ebeam): 
        return self.coord_func(Ebeam, 'x')

    def yfunc(self, Ebeam): 
        return self.coord_func(Ebeam, 'y')
    
    def rfunc(self, Ebeam): 
        return self.coord_func(Ebeam, 'r')

    
class MapScope: 
    def __init__(self): pass

    def full_signal_name(self, alias): pass

    def e_beam(self, shot): pass

    def radref_file(self, shot): pass

    def radref(self): pass

    def plot_camera_and_plasma(self): pass

    def load_sig(self, shot, alias, options): 
        name = self.full_signal_name(alias)
        #name = name % options
        name = add_option(name, options)
        sig = signal(shot, name, device='tjii')
        return sig


class MapInterpolator:
    def process(self, sigdict, E): pass # return sigdict

def nan_array(num, dtype=float): 
    return np.linspace(np.nan, np.nan, num=num, dtype=dtype)

def bin_average(x, y, bins, ymean_func=np.mean, regular_x=False): 
    bin_cnt = len(bins)-1
    _x = nan_array(bin_cnt) #np.empty(bin_cnt, dtype=float)
    _y = nan_array(bin_cnt) #np.empty(bin_cnt, dtype=float)
    
    for i, (x0, x1) in enumerate(zip(bins, bins[1:])): 
        mask = (x >= x0) & (x <= x1)
        if not np.any(mask): continue

        _x[i] = 0.5*(x0 + x1) if regular_x else np.mean(x[mask])
        _y[i] = ymean_func(y[mask])
  
    return _x, _y

def bin_average_sigdict(sigdict, xname, bins, mean_func=np.mean, regular_x=False): 
    bin_cnt = len(bins)-1
    #_sigdict = {k:np.empty(bin_cnt, dtype=float) for k, _ in sigdict.items()}
    _sigdict = {k:nan_array(bin_cnt) for k, _ in sigdict.items()}
    
    x = sigdict[xname]
    for i, (x0, x1) in enumerate(zip(bins, bins[1:])): 
        mask = (x >= x0) & (x <= x1)
        if not np.any(mask): continue

        for k in sigdict.keys(): 
            sig = sigdict[k]
            _sig = _sigdict[k]
            _sig[i] = mean_func(sig[mask])
  
    return _sigdict

def make_rho_signed(uscan, rho): 
    result = rho.copy()
    result = np.abs(result)
    #return result 
    i = np.argmin(result)
       
    if uscan[0] < uscan[-1]: 
        result[0:i+1] *= -1.0
    else:    
        result[i+1:-1] *= -1.0
    return result    

def make_rho_by_xy(x, y, mconf): 
    result = np.zeros_like(x)
    N = len(result)
    for i in range(N): 
        result[i] = mconf.rho(x[i], y[i])

    return result


#%%

map_line_all_names = ['time', 'uscan', 'x', 'y', 'rho', 'itot', 'zd', 'dens', 'test_sig'] # 'rho1'
map_line_loc_names = ['uscan', 'x', 'y', 'rho']            # for localization. Don't need smoothing 
map_line_aux_names = ['time', 'itot', 'zd', 'dens']        # for filtration only
map_line_interp_names = ['uscan', 'x', 'y', 'rho', 'test_sig']   

    
class MapLine: 
    def __init__(self, scope): 
        self.scope = scope
        self.common_options = ''
        self.shots = []
        self.E = None

        for name in map_line_all_names: 
            self.__dict__[name] = None

    def load(self, shot, test_sig, options): 
        self.common_options = options
        self.shots = [shot]
        self.E = self.scope.e_beam(shot)

        self.test_sig = None 
        self.test_sig = self.load_sig(shot, test_sig, resample=False)
        self.time = self.test_sig.x

        for name in map_line_all_names: 
            if name not in ['time', 'test_sig', 'x', 'y', 'rho']:             
                self.__dict__[name] = self.load_sig(shot, name, resample=True)
        
        for name in map_line_all_names: 
            # Signal -> np.array
            sig = self.__dict__[name]
            if isinstance(sig, XYSignal): 
                self.__dict__[name] = sig.y
        

        E = self.scope.e_beam(shot)
        
        radref = self.scope.radref()
        if radref is not None: 
            self.x   = radref.xfunc(E)(self.uscan)
            self.y   = radref.yfunc(E)(self.uscan)
            self.rho = radref.rfunc(E)(self.uscan)
        else: 
            radref_file = self.scope.radref_file(shot)
            rfunc = func_from_file(radref_file, 1, 2, fill_value=np.nan) #fill_value='const')
            xfunc = func_from_file(radref_file, 1, 3, fill_value=np.nan) #fill_value='const')
            yfunc = func_from_file(radref_file, 1, 4, fill_value=np.nan) #fill_value='const')
    
            self.x = xfunc(self.uscan)
            self.y = yfunc(self.uscan)
            self.rho = rfunc(self.uscan)
            
            if np.min(self.x) > 1.0: 
                self.x -= 1.5
            
        
        
    def load_sig(self, shot, name, resample=True): 
        sig = self.scope.load_sig(shot, name, self.common_options)
        if resample: 
            sig.resample_as(self.test_sig)
        return sig

    def refine(self, mapfilter=None, interpolator=None, inplace=False): 
        new_signals = {} 
        processed_names = (map_line_all_names if interpolator is None 
                      else map_line_interp_names)
        
        # filter: all the signals
        processed_names = [name for name in processed_names if self.__dict__.get(name) is not None ]
        
        if mapfilter is not None: 
            bool_mask = mapfilter.calc_mask(name_provider=self)
            for name in processed_names: 
                new_signals[name] = self.__dict__[name][bool_mask]
        else: 
            for name in processed_names: 
                new_signals[name] = self.__dict__[name]
            
            
        # interpolate: only uscan, x, y, test_sig
        if interpolator is not None: 
            new_signals = interpolator.process(new_signals, self.E)
        
        if new_signals is None: 
            return None
        
        if inplace: 
            for name in processed_names: 
                self.__dict__[name] = new_signals[name]
            return self
            
        else: 
            ml = MapLine(self.scope)
            ml.E = self.E
            ml.shots = self.shots
            
            for name in processed_names: 
                ml.__dict__[name] = new_signals[name]
    
            return ml
    
    def upgrade(self, mapfilter=None, interpolator=None): 
        return self.refine(mapfilter, interpolator, inplace=True)

    def asblock(self): 
        if self.uscan is None: 
            return np.zeros([0, 5])

        uu = self.uscan
        xx = self.x
        yy = self.y
        rr = self.rho
        sig = self.test_sig
        
        block = np.vstack((uu, xx, yy, rr, sig)).T
        return block

    def merge(self, other): 
        self.shots.extend(other.shots)        
        
        for name in map_line_all_names: 
            self.__dict__[name] = np.hstack((self.__dict__[name], 
                                            other.__dict__[name]))
            
    def plot(self, x=None, y=None): 
        if self.uscan is None: 
            return 
        
        if x is None: 
            x = self.uscan
        elif x == 'time': 
            x = self.time
        else:     
            x = self.__dict__[x]

        if y is None: 
            y = self.test_sig
        else: 
            y = self.__dict__[y]
                
        plt.plot(x, y)
        
#%%


class MapSet: 
    def __init__(self, scope): 
        self.scope = scope
        self.test_sig_name = None
        self.lines = {}
        
    def loadfromfile(self, filename): 
        with open(filename, 'rb') as f:
             mset = pickle.load(f)
        
        self.scope = mset.scope
        self.test_sig_name = mset.test_sig_name
        self.lines = mset.lines

    def savetofile(self, filename):     
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        

    def load(self, shots, test_sig_name, options, filename=None, recalc=False): 
        if (filename is not None) and os.path.exists(filename) and (not recalc):
            self.loadfromfile(filename)
            return
        
        self.lines = {}
        self.test_sig_name = test_sig_name
        for shot in shots: 
            E = self.scope.e_beam(shot)
            print('shot #{}, E={}'.format(shot, E))
            line = MapLine(self.scope)
     
            try: 
                line.load(shot, test_sig_name, options=options)            
            except BaseException as e: 
                #raise
                if isinstance(e, KeyboardInterrupt): 
                    raise
                
                print('ERROR: ', type(e), e)
                continue 
            
            if not E in self.lines.keys(): 
                self.lines[E] = line
            else: 
                self.lines[E].merge(line)
        
        if filename is not None: 
            self.savetofile(filename)


    def refine(self, mapfilter=None, interpolator=None, inplace=False): 
        if inplace: 
            for E, line in self.lines.items(): 
                line.refine(mapfilter, interpolator, inplace=True)
            return self
        else: 
            result = MapSet(self.scope)
            result.test_sig_name = self.test_sig_name
            for E, line in self.lines.items(): 
                line_E = line.refine(mapfilter, interpolator)
                if line_E is not None: 
                    result.lines[E] = line_E
            
            return result

    def upgrade(self, mapfilter=None, interpolator=None): 
        return self.refine(mapfilter, interpolator, inplace=True)
        

    def assemble(self): 
        agglomerate = np.zeros([0, 5])  # uscan, x, y, rho, signal
        for E, line in self.lines.items(): 
            block = line.asblock()
            agglomerate = np.vstack((agglomerate, block))
        return agglomerate

    def plot1d(self, x=None, y=None): 
        plt.figure()
        for E, line in self.lines.items(): 
            line.plot(x, y)

    def data2d(self): 
        agglomerate = self.assemble()
        # [0]uscan [1]x [2]y [3]rho [4]Phi 
        sig = agglomerate[:,4]
        sigXY = agglomerate[:,1:3]
        
        xmin, xmax = -0.35, 0.1
        ymin, ymax = -0.35, 0.1
        limits = [xmin, xmax, ymin, ymax]
        
        Npoints = 50 #100 # 50
        
        x, y = np.linspace(xmin, xmax, Npoints), np.linspace(ymin, ymax, Npoints)
        meshx, meshy = np.meshgrid(x,y)
        
        return sig, sigXY, meshx, meshy, limits
  
    def plot2d(self, kind='both'): 
        sig, sigXY, meshx, meshy, limits = self.data2d()

        min_val, max_val = np.nanmin(sig), np.nanmax(sig)
        norm = cm.colors.Normalize(vmax=max_val, vmin=min_val)
        cmap = plt.get_cmap('jet')
    
        if kind in ('both', 'interp'): # plot 2D
            sig_grid = interpolate.griddata(sigXY, sig, (meshx, meshy), fill_value=np.nan, method='linear')    
            plt.figure()
            plt.imshow(sig_grid, interpolation='none', origin='lower', extent=limits, norm=norm, cmap=cmap)
            self.scope.plot_camera_and_plasma()
            
        if kind in ('both', 'exp'): # plot experimental curves
            plt.figure()
            mask = sig > -100.0  #??? sig can be nan ???
            plt.scatter(sigXY[mask,0], sigXY[mask, 1], c=sig[mask], s=100, norm=norm, cmap=cmap) 
            self.scope.plot_camera_and_plasma()        

        
    def save2d(self, filename):
        sig, sigXY, meshx, meshy, limits = self.data2d()
        sig_grid = interpolate.griddata(sigXY, sig, (meshx, meshy), fill_value=np.nan, method='linear')
        #with open(filename, 'w') as f:
        xx = meshx[0,:]
        yy = meshy[:,1]
        data = np.vstack((xx, sig_grid))
        yy = np.insert(yy, 0, 0.0)
        L = len(yy)
        yy = yy.reshape(L, 1)
        data = np.hstack((yy, data))

        np.savetxt(filename, data, fmt='%.2e')
        
    
    
#%%
        
if __name__ == '__main__':
    pass

