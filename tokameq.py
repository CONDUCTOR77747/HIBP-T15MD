# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:52:01 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

#%%

def get_named_param(s, name, dtype=float): 
    i = s.find(' ' + name + ':')
    if i == -1: 
        return None
    s_after_name = s[i+len(name)+2:].strip()
    s_val = s_after_name.split(' ')[0]
    return dtype(s_val)


def find_str_in_lines(lines, s): 
    for i in range(len(lines)):
        if lines[i].strip() == s:
            return i
    return -1

def array_from_str(s, skip=0): 
    try: 
        arr = [float(x) for x in s.strip().split()[skip:]]
        return np.array(arr)
    except ValueError: # ValueError: could not convert string to float: '-7.29513e-050.000843813'
        # 12 chars per value  +  indent of 6 chars in the beginning
        s = s[6:]  # skip indent
        ss =  [ s[i: i+12] for i in range(0, len(s), 12) ]
        arr = [ float(x) for x in ss if x.strip() != '' ]
        return np.array(arr)

def array2D_from_lines(lines): 
    arr = None

    for s in lines: 
        if s.strip() == '': 
            break
        if arr is None: 
            arr = array_from_str(s)
        else: 
            _arr = array_from_str(s)
            arr = np.vstack((arr, _arr))

    return arr

def plot_point(pt, *args, **kwargs): 
    xx = [pt[0]] 
    yy = [pt[1]] 
    plt.plot(xx, yy, *args, **kwargs) # plt.plot(xx, yy, 'o', **kwargs) markersize

def plot_points(xx, yy, zz): 
    #cmap = mpl.colormaps['jet']
    cmap = plt.get_cmap('jet', 1000.0)

    maxz = np.max(zz)
    #minz = np.min(zz)
    #_maxz = max(abs(maxz), abs(minz))

    for i, x in enumerate(xx): 
        for j, y in enumerate(yy): 
            z = zz[j, i]/maxz
            if z < 0.0000001: continue

            ms = int(z * 10)
            c = cmap(int(z*1000.0))
            plot_point([x, y], 'o', ms=ms, color=c)

#    for i, x in enumerate(self.xx): 
#        for j, y in enumerate(self.yy): 
#            v = self.values[j, i]/2.0/_maxv
#
#            ms = int(abs(v) * 12 )
#            if ms > 20: print(v, ms, self.values[j, i], _maxv, minv, maxv)
#            c = cmap(int(v*1000 + 500))
#
#            plot_point([x, y], 'o', ms=ms, color=c)

#%%

def onWhealEvent(event):
    if event.name == 'scroll_event': 
        ax = plt.gca()
        cmin, cmax = ax.images[0].get_clim()

        if event.button == 'down':  k = 1.1
        elif event.button == 'up':  k = 0.9

        cmax *= k
        #if not abs(cmin)*10 < abs(cmax): cmin *= k
        if not abs(cmin)*10 < abs(cmax): 
            cmin = -cmax
        
        ax.images[0].set_clim((cmin, cmax))        
        plt.draw()

class TokameqCoil: 
    # NN      Position (r,z)      Value       Normal. val.    "Thickness" (DR,DZ,NR,NZ)
    # 1       0.335     1.845     -0.45         -0.45          0.132 0.874     7    35   
    def __init__(self, data): 
        self.num = int(data[0])
        self.r       = data[1]     # x 
        self.z       = data[2]     # y
        self.I       = data[3] 
        self.normI   = data[4] 
        self.dr      = data[5] 
        self.dz      = data[6] 
        self.nr  = int(data[7])
        self.nz  = int(data[8]) 

    def get_points(self): 
        for i in range(self.nr): 
            for j in range(self.nz): 
                rh = self.dr/(self.nr - 1)
                zh = self.dz/(self.nz - 1)
                r = self.r - self.dr*0.5 + i*rh
                z = self.z - self.dz*0.5 + j*zh
                yield np.array([r, z])
            

    def plot(self, shownumbers=False): 
        xmin = self.r - self.dr*0.5
        ymin = self.z - self.dz*0.5
        
        plt.gca().add_patch(Rectangle((xmin, ymin), self.dr, self.dz, 
                                       linewidth=1, edgecolor='tab:gray',
                                       facecolor='tab:gray'))
        if shownumbers: 
            plt.text(self.r, self.z, str(self.num))

    def plot_points(self):
        for pt in self.get_points(): 
            plot_point(pt, 's', ms=1, color='black')

class TokameqCoils: 
    def __init__(self, lines): 
        start_section = find_str_in_lines(lines, 'External currents:')
        data = array2D_from_lines(lines[start_section+2:])
        self.coils = []
        for i in range(data.shape[0]): 
            coil = TokameqCoil(data[i, :])
            self.coils.append(coil)
      
    def plot(self, shownumbers=False): 
        for c in self.coils: 
            c.plot(shownumbers)

    def plot_points(self):
        for c in self.coils: 
            c.plot_points()
            
class TokameqData2D: 
    def __init__(self, lines, section_title): 
        start_section = find_str_in_lines(lines, section_title)

        self.rr = array_from_str(lines[start_section+1], skip=1) 
        data = array2D_from_lines(lines[start_section+2:])
        self.zz = data[:, 0]
        self.values = data[:, 1:]
    
    def plot(self): 
        #kwargs = {'cmap': cm.jet, 'aspect': 'auto', 'interpolation': 'bilinear', 'origin': 'lower'}
        kwargs = {'cmap': cm.jet, 'aspect': 'auto', 'interpolation': 'none', 'origin': 'lower'}

        vv = self.values[::-1, :]
        plt.imshow(vv, extent = [self.rr[0], self.rr[-1], self.zz[0], self.zz[-1]], **kwargs)
        self.scroll_event_id = plt.gcf().canvas.mpl_connect('scroll_event', onWhealEvent)

        plt.colorbar()
        #plt.axis('equal')
        plt.xlabel('r (m)')
        plt.ylabel('z (m)')


    def plot_points(self): 
        plot_points(self.rr, self.zz, self.values)


#%%

class TokameqFile: 
    def __init__(self, filename): 
        with open(filename, 'r') as f:
            self.lines = f.readlines()

        self.nr = get_named_param(self.lines[2], 'R') 
        self.nz = get_named_param(self.lines[2], 'Z') 

        self.coils = TokameqCoils(self.lines)
        self.Jpl = TokameqData2D(self.lines, 'Current density J(r,z)')
        self.F = TokameqData2D(self.lines, 'Poloidal flux F(r,z)')
        self.Hr = TokameqData2D(self.lines, 'Magnetic field r component Hr(r,z)')
        self.Hz = TokameqData2D(self.lines, 'Magnetic field z component Hz(r,z)')


#%%

if __name__ == '__main__': 
    #tok = TokameqFile(r'D:\radrefs\HIBP-T15MD-master\1MA_sn.txt')
    tok = TokameqFile(r'1MA_sn.txt')
    
    #tok.F.plot()
    tok.Jpl.plot()
    
    plt.figure()
    tok.coils.plot(True)
    tok.coils.plot_points()
    tok.Jpl.plot_points()

    plt.figure()
    tok.Hz.plot()

