# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:24:37 2019

@author: reonid

   TJ-II Magnetic configuration
   sorts, smoothes and regularizes the surface arrays
   
   mconf = MagConf("J:\\reonid\\Regimes\\100_44_64")
   mconf.plot(color='k', regular=10)
   rho = mconf.rho((0.1, 0.2))
   qangle = mconf.quasi_angle((0.1, 0.2))
   sf = mconf.surface_at_rho(0.5)
   L = mconf.surface_length(0.5)
   x, y, gdata = mconf.griddata('rho', GRID_SIZE)
   
"""


#from copy import deepcopy
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter

from .geom.geomx import vAbs
from .geom.geom2d import Polygon2D, Ray2D, Segment2D, vector2D #, vAbs


SURF_LEN = 1000 
GRID_SIZE = 500

class MagSurf: 
    def __init__(self, center, xarray, yarray, artificial=False): 
        self.center = center
        self.artificial = artificial
        if artificial:
            self.x, self.y = xarray, yarray
            self.raw_x, self.raw_y = xarray, yarray
            self.sorted_x, self.sorted_y = xarray, yarray  # all the same
        else: 
            self.x, self.y = xarray, yarray
            self.raw_x, self.raw_y = xarray.copy(), yarray.copy()
            self.x, self.y = sort_contour(self.x, self.y)
   
        self.area = None
        self._broken = None
        self.rho = None
        self.norm_area = None
        
        #self.enhance(smooth)
    
    def zipxy(self):
        return zip(self.x, self.y)

    def broken(self): 
        if self._broken is not None: 
            return self._broken
        else: 
            shifted_x = np.roll(self.x, -1)
            shifted_y = np.roll(self.y, -1)
            pairwise_dist = np.hypot(shifted_x - self.x, shifted_y - self.y)
            
            self._broken = np.max(pairwise_dist) > 0.06   # 0.03:OK  0.3:broken
            self.length = np.sum(pairwise_dist)
            return self._broken

    def plot(self, kind='smoothed', color='k'): 
        if kind == 'smoothed': 
            plt.plot(self.x, self.y, color)
        elif kind == 'raw': 
            plt.plot(self.raw_x, self.raw_y, color)
        elif kind == 'sorted': 
            plt.plot(self.sorted_x, self.sorted_y, color)
        #plt.plot(self.dist_from_base)
        #plt.plot(self.pairwise_dist)

    def clean(self): 
        L = self.length
        min_d = L/len(self.x)*0.1
        self.x, self.y = sort_contour(self.x, self.y, min_d) #0.0005*5)

    def invert(self): 
        self.x = self.x[::-1]
        self.y = self.y[::-1]
        # first point then same
        self.x = np.roll(self.x, 1)   
        self.y = np.roll(self.y, 1)

    def recalc_metrics(self): 
        shifted_x_minus = np.roll(self.x, -1)
        shifted_y_minus = np.roll(self.y, -1)

        self.pairwise_dist = np.hypot(shifted_x_minus - self.x, shifted_y_minus - self.y)
        self.length = np.sum(self.pairwise_dist)
        
        if self.length > 0.0: 
            K = 15 + int(1.0/self.length)
        else:
            K = 15
            
        self.poloidal_dx = np.roll(self.x, +K) - np.roll(self.x, -K)
        self.poloidal_dy = np.roll(self.y, +K) - np.roll(self.y, -K)


        self.dist_from_base = np.zeros_like(self.x)
        self.dist_from_base[0] = 0.0
        for i, d in enumerate(self.pairwise_dist): 
            if i > 0: 
                self.dist_from_base[i] = self.dist_from_base[i-1] + d

        pgn = Polygon2D(self.zipxy())
        self.area = pgn.area()

        self.norm_dist_from_base = self.dist_from_base/self.length
        self.norm_area = None # calculated later in MagConf

   
    def enhance(self, smooth=None):
        if self.artificial: 
            self.recalc_metrics()
            return 
        
        self.recalc_metrics()
        self.clean()
        self.recalc_metrics()

        # first point of contour should be on the line 
        # that connects center and hardcore point (0, 0)

        x0, y0 = self.center
        ang0 = np.arctan2(0.0-y0, 0.0-x0)
        
        x = self.x[-1]
        y = self.y[-1]
        prev_ang = np.arctan2(y-y0, x-x0)
        for i, (x, y) in enumerate(self.zipxy()): 
            ang = np.arctan2(y-y0, x-x0)
            d = ang - ang0
            prev_d = prev_ang - ang0
            if np.sign(d) != np.sign(prev_d): 
                if abs(d) < 0.5 and abs(prev_d) < 0.5: 
                    sgm = Segment2D((self.x[i-1], self.y[i-1]), (x, y))
                    ray = Ray2D((x0, y0), (-x0, -y0))
                    pt = sgm & ray # intersection point
                    
                    ii = i if abs(d) < abs(prev_d) else i-1 # nearest
                    self.x[ii] = pt[0]
                    self.y[ii] = pt[1]
                    self.x = np.roll(self.x, -ii)   
                    self.y = np.roll(self.y, -ii)

            prev_ang = ang
        
        # all contours should be oriented the same way
        if self.x[5] > self.x[0]: 
            self.invert()    
        
        self.sorted_x = self.x.copy()
        self.sorted_y = self.y.copy()
        
        if smooth is not None: 
            self.smooth(smooth)  
            
        if self.x[5] > self.x[0]: 
            self.invert()  
            self.recalc_metrics()
            
        #self.recalc_metrics()

    def _smooth(self, smooth, sort): 
        self.x = np.r_[self.x, self.x[0]]
        self.y = np.r_[self.y, self.y[0]]

        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([self.x, self.y], k=3, s=smooth, per=True)

        # evaluate the spline fits for SURF_LEN evenly spaced distance values
        self.x, self.y = interpolate.splev(np.linspace(0, 1, SURF_LEN), tck)
        if sort: 
            self.x, self.y = sort_contour(self.x, self.y, 0)

        self.recalc_metrics() 

    def smooth(self, smooth=0.00001):   
        if self.artificial: 
            return 

        # interpolation can give narrow loops !!!
        self._smooth(smooth, sort=True)      # sort can change the length 
        
        # next iteration usually gives result without loops
        self._smooth(smooth, sort=False)     # len(self.x) should be SURF_LEN here !


    def index_of_nearest(self, x, y):
        distances = np.hypot(self.x - x, self.y - y)
        return np.argmin(distances)
    
    def index_of_nearest_ang(self, x, y):
        a = np.arctan2(y - self.center[1], x - self.center[0])
        angles = np.arctan2(self.y - self.center[1], self.x - self.center[0])
        da = angles - a
        return np.argmin(da)

            
#%%
        
class MagConf: 
    def __init__(self, filepath, smooth=0.00001): 

        self.surfaces = []
        
        if filepath.endswith(".txt"): 
            self.loadtxt(filepath)
            #self.center = self.surfaces[0].center
            #self.enhance(smooth)
            #self.recalc_grids()
        else:
            self.loadpath(filepath)
            #self.center = self.surfaces[0].center
            self.enhance(smooth)
            self.recalc_grids()
    
    
    def loadtxt(self, filename): 
        #data = np.loadtxt("c:\\reonid\\2021\\100_44_64.txt", dtype="float")
        data = np.loadtxt(filename, dtype="float")
        
        xs = data[:, 0]
        ys = data[:, 1]
        
        ii = np.argwhere(np.isnan(ys)).squeeze()
        rs = xs[ii]
        
        slices = [slice(i0+1, i1) for i0, i1 in zip(ii[:-1], ii[1:])]
        
        for slc, r in zip(slices, rs): 
            xx = xs[slc]
            yy = ys[slc]
            x0 = np.mean(xx)
            y0 = np.mean(yy)
            sf = MagSurf((x0, y0), xx, yy, True)
            sf.rho = r
            self.surfaces.append(sf)

        self.center = self.surfaces[0].center

    def loadpath(self, filepath): 
        for i in range(71, 76):   # there is a contour of TJ-II in fort.70 
            filename = '%s\\fort.%d' % (filepath, i)
            self.load_surfaces_from(filename)
        
        self.surfaces = [sf for sf in self.surfaces if not sf.broken()]
        self.surfaces = self.surfaces[::-1]
        self.center = self.surfaces[0].center

    def load_surfaces_from(self, filename): 
        if not os.path.exists(filename): 
            return   # OK
        
        with open(filename) as f:
            text = f.readlines()
            
            col_titles = text[0].split()                          # 1st line: X Z X2 Z2 ...
            center = tuple( float(s) for s in text[1].split() )   # 2nd line: center 
            ncols = len(col_titles)
            
            cols = [[] for i in range(ncols)]
            
            for line in text[2:]: 
                for i, s in enumerate(line.split()): 
                    cols[i].append(s)
            
            for i in range(ncols // 2): 
                x_idx = i*2
                y_idx = i*2 + 1
                
                xx = np.array([float(s) for s in cols[x_idx]])
                yy = np.array([float(s) for s in cols[y_idx]])
                
                sf = MagSurf(center, xx, yy)
                #sf.sort()
                self.surfaces.append(sf)

    def plot(self, kind='smoothed', color='k', regular=None): 
        if kind == 'rho': 
            plt.imshow(self.rho_grid.T, interpolation='none', origin='lower', 
                       extent = self.limits)
        elif kind == 'quasi_angle': 
            plt.imshow(self.quasi_angle_grid.T, interpolation='none', origin='lower', 
                       extent = self.limits)
        else: 
            if regular is not None: 
                for i in range(regular+1): 
                    rho = i/regular
                    if rho == 0: rho = 0.001
                    sf = self.surface_at_rho(rho)
                    sf.plot(kind, color)
            else: 
                for sf in self.surfaces: 
                    sf.plot(kind, color)

    def enhance(self, smooth=0.00001): 
        for sf in self.surfaces: 
            sf.enhance(smooth)
        
        self.unit_area = self.surfaces[-1].area
        self.recalc_metrics()   
        
        sf001 = self.surface_at_rho(0.001)
        self.surfaces.insert(0, sf001)
        self.recalc_metrics()

        sf105 = self.surface_at_rho(1.05)
        self.surfaces.append(sf105)
        self.recalc_metrics()
        
    def recalc_metrics(self): 
        for sf in self.surfaces: 
            sf.norm_area = sf.area / self.unit_area
            sf.rho = sf.norm_area ** 0.5

    def recalc_grids(self): 
        x, y, self.rho_grid = self.griddata('rho', GRID_SIZE)
        x, y, self.quasi_angle_grid = self.griddata('quasi_angle', GRID_SIZE)
        self.grid_mesh = (x, y)
        self.limits = [x[0], x[-1], y[0], y[-1]]
            
    def calc_limits(self): 
        last_sf = self.surfaces[-1]
        xmin = np.min(last_sf.x)
        xmax = np.max(last_sf.x)
        ymin = np.min(last_sf.y)
        ymax = np.max(last_sf.y)
        return xmin, ymin, xmax, ymax
    
    def griddata(self, name, N, func='griddata', **kwargs): 
        xx = np.array([], dtype='float64')
        yy = np.array([], dtype='float64')
        rr = np.array([], dtype='float64')
        aa = np.array([], dtype='float64')
        
        for sf in self.surfaces: 
            x = sf.x
            y = sf.y
            r = sf.rho*np.ones_like(sf.y)
            a = sf.norm_dist_from_base*np.pi*2.0
            
            xx = np.hstack((xx, x))
            yy = np.hstack((yy, y))
            rr = np.hstack((rr, r))
            aa = np.hstack((aa, a))

        if name == 'rho': 
            values = rr
        elif name == 'quasi_angle':  # quasi-angle
            values = aa

        xxyy = np.vstack((xx, yy)).T

        xmin, ymin, xmax, ymax = self.calc_limits() 
        x = np.linspace(xmin, xmax, N) 
        y = np.linspace(ymin, ymax, N)
        xmesh, ymesh = np.meshgrid(x, y)
        
        if func == 'griddata': 
            gdata = interpolate.griddata(xxyy, values, (xmesh, ymesh), fill_value=np.nan, method='cubic', **kwargs)  # 'linear'
            # gdata: y - 1st index, x - 2nd index
            return x, y, gdata.T 

        if func == 'interp2d': 
            interp = interpolate.interp2d(xx, yy, values, fill_value=np.nan, kind='cubic', **kwargs)  # 'linear'
            return interp

    def surface_at_rho(self, rho): 
        N = len(self.surfaces)
        rhos = []
        for sf in self.surfaces: 
            rhos.append(sf.rho)

        i0, i1, t = search_sorted_ex(rhos, rho)
        x0, y0 = self.center

        if i1 == 0: 
            rho0 = self.surfaces[0].rho
            t = rho/rho0
            xx = (self.surfaces[0].x - x0)*t + x0
            yy = (self.surfaces[0].y - y0)*t + y0
            
        elif i0 >= N-1: 
            t = (rho - self.surfaces[-1].rho)/(self.surfaces[-2].rho - self.surfaces[-1].rho)
            xx = self.surfaces[-1].x + (self.surfaces[-2].x - self.surfaces[-1].x)*t
            yy = self.surfaces[-1].y + (self.surfaces[-2].y - self.surfaces[-1].y)*t
        else: 
            xx = self.surfaces[i0].x + (self.surfaces[i1].x - self.surfaces[i0].x)*t 
            yy = self.surfaces[i0].y + (self.surfaces[i1].y - self.surfaces[i0].y)*t            

        sf = MagSurf(self.center, xx, yy, artificial=True)
        sf.recalc_metrics()
        return sf

    def rho_of_pt(self, point): 
        return value_at(point, self.grid_mesh, self.rho_grid)

    def rho(self, x, y): 
        return value_at((x, y), self.grid_mesh, self.rho_grid)

    def quasi_angle(self, point): 
        return value_at(point, self.grid_mesh, self.quasi_angle_grid)

    def surface_length(self, rho): 
        sf = self.surface_at_rho(rho)
        return sf.length


#%% sort curve

class EndOfPoints(Exception): 
    pass

def find_nearest(idx, xarray, yarray, flags):
    flags[idx] = 1
    
    if np.all(flags): 
        raise EndOfPoints
    
    x, y = xarray[idx], yarray[idx]
    distances = np.hypot(xarray - x, yarray - y)
    distances[flags] = np.inf
    i_min = np.argmin(distances)
    flags[i_min] = 1 
    return i_min, idx

def sort_contour(xarray, yarray, min_dist=0.0): 
    '''
    sorts points of any smooth contour without large gaps
    '''
    flags = np.zeros_like(xarray, dtype='bool')  # already processed points

    i = 0
    sorted_x = [xarray[i]]
    sorted_y = [yarray[i]]
    while True: 
        try: 
            i, iprev = find_nearest(i, xarray, yarray, flags)
            d = np.hypot(xarray[i]-xarray[iprev], yarray[i]-yarray[iprev])
            if (d > min_dist): # or min_dist == 0.0:
                sorted_x.append(xarray[i])
                sorted_y.append(yarray[i])
            else: 
                # too close points. 
                sorted_x[-1] = 0.5*(xarray[i] + xarray[iprev])
                sorted_y[-1] = 0.5*(yarray[i] + yarray[iprev])
                i = iprev
        except EndOfPoints: 
            break

    return np.array(sorted_x), np.array(sorted_y)


#%%  utilities

def search_sorted_ex(array, value): 
    L = len(array)
    i = np.searchsorted(array, value)
    '''
    try: 
        j = i[0]
        oldi, i = i, j
        print(type(oldi))
    except: 
        pass       
    '''
    if i <= 0: 
        return (0, 0, 1.0)
    elif i >= L: 
        return (L-1, L-1, 0.0)
    else: 
        v0 = array[i-1]
        v1 = array[i]
        t = (value-v0)/(v1-v0)
        return (i-1, i, t)

def value_at(point, xydata, zdata): 
    x, y = point
    x1d, y1d = xydata 
    z2d = zdata
    i0, i1, tx = search_sorted_ex(x1d, x)
    j0, j1, ty = search_sorted_ex(y1d, y)

    z00 = z2d[i0, j0]
    z01 = z2d[i0, j1]
    z10 = z2d[i1, j0]
    z11 = z2d[i1, j1]

    z00_10 = z00 + (z10-z00)*tx;
    z01_11 = z01 + (z11-z01)*tx;
    return z00_10 + (z01_11 - z00_10)*ty;
  

#%%  test    

def radial_dist(mconf, pt0, pt1): 
    ''' 
    calc radial dist    
    '''
    r0 = mconf.rho(pt0)
    r1 = mconf.rho(pt1)
    result = r1 - r0
    if np.isnan(result): 
        return 0.0
    else: 
        return result

def poloidal_dist(mconf, pt0, pt1, absval=False): 
    ''' 
    calc linear distance 
    in case of small distances OK
    ??? in principle, we need curvilinear distance    
    
    '''

    pt = 0.5*(pt0 + pt1)
    r0 = mconf.rho_of_pt(pt0)
    r1 = mconf.rho_of_pt(pt1)
    #rmean = (r0+r1)*0.5
    #rmax = max(r0, r1)
    if r1 > r0:
        rmax = r1
        pt = pt1
    else:
        rmax = r0
        pt = pt0
        
     
    if rmax > 0.1: 
        sf = mconf.surface_at_rho(rmax) # 
        idx = sf.index_of_nearest(pt[0], pt[1]) 
    else:
        sf = mconf.surface_at_rho(0.1) 
        idx = sf.index_of_nearest_ang(pt[0], pt[1]) 
        
    
    pol_dx = sf.poloidal_dx[idx]
    pol_dy = sf.poloidal_dy[idx]
    
    pol_vect = vector2D(pol_dx, pol_dy)
    norm_pol_vect = pol_vect / vAbs(pol_vect)
    sv_vect = vector2D(pt1[0] - pt0[0], pt1[1] - pt0[1])
    result = sv_vect.dot(norm_pol_vect)
    
    #result = np.arctan2(pol_dx, pol_dy)
     
    if np.isnan(result): 
        return 0.0
    else: 
        if absval: result = abs(result)
        return result    

if __name__ == '__main__':
    
    m = MagConf('J:\\reonid\\Regimes\\58_83_61')
    m.plot('rho')
    #m = MagConf('J:\\reonid\\Regimes\\100_44_64')
    #m.plot('quasi_angle')

    '''
    x, y, gdata = m.griddata('quasi_angle', 1000)
    #x, y, gdata = m.griddata('rho', 1000)
    #plt.imshow(gdata, interpolation='none', origin='lower')

     
    gdata = gaussian_filter(gdata, sigma=(2.0,2.0), mode='nearest')
    gr = np.gradient(gdata, edge_order=2)

    ang = np.arctan2(gr[1], gr[0])
    #plt.imshow(gr[0], interpolation='none', origin='lower')
    plt.imshow(ang, interpolation='none', origin='lower')
    '''


