# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:06:43 2023


??? look on following !!!
interpn
    a convenience function which wraps RegularGridInterpolator


scipy.ndimage.map_coordinates
    interpolation on grids with equal spacing (suitable for e.g., N-D image resampling)


@author: reonid

_numba_njit = numba.njit

"""

import numpy as np
import copy
import time
import sys
import os

from copy import deepcopy

#import builtins
#print = builtins.print

import matplotlib.pyplot as plt

#from scipy.interpolate import RegularGridInterpolator
#from scipy.ndimage import map_coordinates

import hibplib as hb
import define_geometry as defgeom

import hibpcalc.geomfunc as gf
import hibpcalc.misc as misc
import hibpcalc.fields as fields

from hibpcalc.fields import __return_E as return_E
from hibpcalc.geomfunc import vNorm

#import hibpplotlib as hbplot
#import define_geometry as defgeom

#import hibplib.physconst as physconst

PI = 3.1415926535897932384626433832795
SI_AEM     = 1.6604e-27       #     {kg}
SI_e       = 1.6021e-19       #     {C}
SI_Me      = 9.1091e-31       #     {kg}      // mass of electron
SI_Mp      = 1.6725e-27       #     {kg}      // mass of proton
SI_c       = 2.9979e8         #     {m/sec}   // velocity of light
SI_1eV     = SI_e             #     {J}
SI_1keV    = SI_1eV*1000.0    #     {J}

MAX_TRAJ_LEN = 5000


Ipl = 1.5  # MA
Btor = 1.0 # T


# %% Define Geometry
geomT15 = defgeom.define_geometry(analyzer=1)

E_slow, E_fast = {}, {}

# load E for primary beamline
fields._read_plates('prim', geomT15, E_slow, E_fast, hb.createplates)
fields._read_plates('sec',  geomT15, E_slow, E_fast, hb.createplates)

# add diafragm for A3 plates to Geometry
hb.add_diafragm(geomT15, 'A3', 'A3d', diaf_width=0.05)
hb.add_diafragm(geomT15, 'A4', 'A4d', diaf_width=0.05)

an_G = geomT15.plates_dict['an'].G
# add detector coords to dictionary
an_edges = geomT15.plates_dict['an'].det_edges
geomT15.r_dict['det'] = an_edges[ an_edges.shape[0]//2 ] [0] 


pf_coils = fields.import_PFcoils('PFCoils.dat')
PF_dict = fields.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)

Binterp = fields.read_B_new(Btor, Ipl, PF_dict, dirname='magfield')
Binterp.cure_artefacts_from_filaments()

#U_dict = {'A2': 50.0, 'B2': -4.0, 'A3': 3.0, 'B3': 4.0, 'A4': 5.0, 'an': 107.0}
#U_dict = {'A2': 14.0, 'B2': 0.0, 'A3': 0.4, 'B3': 0.3, 'A4': 0.1, 'an': 50.0}

#%%

class Efield: 
    def __init__(self, E_fast, U, geom): 
        self.E_fast = E_fast
        self.U = U
        self.geom = geom

    def __call__(self, r): 
        return return_E(r, self.E_fast, self.U, self.geom)

class Efield0: 
    def __init__(self): 
        pass

    def __call__(self, r): 
        return np.zeros(3)

#%%
        
class Bfield: 
    def __init__(self, Binterp): 
        self.Binterp = Binterp

    def __call__(self, r): 
        return self.Binterp(r)[0]

#%%

def f(k, E, V, B):
    return k*(E + np.cross(V, B))


def g(V):
    return V


def _runge_kutt(k, RV, dt, E, B):
    '''
    Calculate one step using Runge-Kutta algorithm

    V' = k(E + [VxB]) == K(E + np.cross(V,B)) == f
    r' = V == g

    V[n+1] = V[n] + (h/6)(m1 + 2m2 + 2m3 + m4)
    r[n+1] = r[n] + (h/6)(k1 + 2k2 + 2k3 + k4)
    m[1] = f(t[n], V[n], r[n])
    k[1] = g(t[n], V[n], r[n])
    m[2] = f(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    k[2] = g(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    m[3] = f(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    k[3] = g(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    m[4] = f(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])
    k[4] = g(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])

    Parameters
    ----------
    k : float
        particle charge [Co] / particle mass [kg]
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        coordinates and velocities array [m], [m/s]
    dt : float
        timestep [s]
    E : np.array([Ex, Ey, Ez])
        values of electric field at current point [V/m]
    B : np.array([Bx, By, Bz])
        values of magnetic field at current point [T]

    Returns
    -------
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        new coordinates and velocities

    '''
    
    #r = RV[0, :3]
    #V = RV[0, 3:]

    r = RV[:3]
    V = RV[3:]

    m1 = f(k, E, V, B)
    k1 = V # g(V)

    fV2 = V + (dt / 2.) * m1
    gV2 = V + (dt / 2.) * m1
    m2 = f(k, E, fV2, B)
    k2 = gV2 #g(gV2)

    fV3 = V + (dt / 2.) * m2
    gV3 = V + (dt / 2.) * m2
    m3 = f(k, E, fV3, B)
    k3 = gV3 # g(gV3)

    fV4 = V + dt * m3
    gV4 = V + dt * m3
    m4 = f(k, E, fV4, B)
    k4 = gV4 # g(gV4)

    V = V + (dt / 6.) * (m1 + (2. * m2) + (2. * m3) + m4)
    r = r + (dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4)

    RV = np.hstack((r, V)) 
    #RV = np.r_[r, V]
    return RV #[0, :]


#  fail gabarit oversize
class Trajectory():
    def __init__(self, q, m, Ebeam, rv0, dt=1e-7):  # U, 
        self.dt = dt
        self.q = q
        self.m = m
        self.t_max = 1e-5
        self.Ebeam = Ebeam # Nominal
        self.rrvv = rv0.reshape(1, 6) # None # np.empry((0, 6))
        #self.rv0 = rv0

        self.check_energy()
        
        self.obstacle = None  
        self.success = None
        self.fan = [] 
        self.secondary = None
        
        self.r_intersect = None
        #self.index = None 
        
    @classmethod
    def from_injector(cls, q, m, Ebeam, r0, alpha, beta, dt=1e-7):  # U, 
        Ebeam_J = Ebeam * 1.602176634E-16
        v_abs = np.sqrt(2.0 * Ebeam_J / m)
        v0 = gf.calc_vector(-v_abs, alpha, beta)
        rv0 = np.hstack( (r0, v0) )

        tr = Trajectory(q, m, Ebeam, rv0, dt)
        return tr        

    def check_energy(self): 
        v_abs = vNorm(self.rrvv[0, 3:6])
        _Ebeam_keV = 0.5*self.m*v_abs*v_abs / 1.602176634E-16 
        if abs(self.Ebeam - _Ebeam_keV) > 0.5: #???
            raise Exception("Trajectory: Ebeam is not consistent with the velocity %f %f" % (self.Ebeam, _Ebeam_keV) )
        elif abs(self.Ebeam - _Ebeam_keV) > 0.01: 
            pass
            #print("Ebeam = %f (%f)" % (Ebeam, _Ebeam_keV) )    
        
    def run(self, E, B, stop): # run
        q_m = self.q / self.m
        rv  = self.rrvv[-1] 
        dt = self.dt
        t = 0.0 # ??? if resume ?  

        _rrvv = np.empty( (MAX_TRAJ_LEN, 6) ); _curr = 0
        #_rrvv[_curr] = rv; _curr += 1 
 
        while True: # t < self.t_max: 
            r = rv[0:3]
            E_loc = E(r)
            B_loc = B(r)
            if np.any(np.isnan(B_loc)): 
                self.obstacle = 'B_is_nan' # ??? 'out'
                break
        
            #rv_new = hb.runge_kutt(q_m, rv, dt, E_loc, B_loc)
            rv_new = _runge_kutt(q_m, rv, dt, E_loc, B_loc)
            _rrvv[_curr] = rv_new; _curr += 1 
            if _curr >= _rrvv.shape[0]: 
                 _rrvv = np.vstack((   _rrvv, np.empty( (MAX_TRAJ_LEN, 6) )  ))
                 print('long traj: _rrvv.shape= ', _rrvv.shape)
            
            if stop(self, rv, rv_new): 
                break
            
            rv = rv_new
            t = t + dt
        else: 
            self.obstacle = 't_max_expired' # ???  "out"
        
        #self.rrvv = _rrvv[0:_curr]
        self.rrvv = np.vstack((  self.rrvv, _rrvv[0:_curr]  ))
    
    # def resume(self):  # continue

    def plot(self, *args, **kwargs): 
        xx = self.rrvv[:, 0] 
        yy = self.rrvv[:, 1] 
        plt.plot(xx, yy, *args, **kwargs)
        
    def cut_with_plane(self, plane):
        plane_r, plane_n = plane
        for r0, r1 in self.segments():
            r_intersect = gf.plane_segment_intersect(plane_n, plane_r, r0, r1) 
            if r_intersect is not None: 
                return r_intersect
        return None 
    
    def trim(self, last_rv): 
        idx = self.find(last_rv[0:3])
        if idx is not None: 
            tr = deepcopy(self)
            tr.rrvv = tr.rrvv[0:idx+1]
            return tr
    
    def segments(self): 
        yield from zip(self.rrvv[0:-1, 0:3], self.rrvv[1:, 0:3])

    def find(self, r): 
        dd = self.rrvv[:, 0:3] - r
        idx = np.argmin(vNorm(dd, axis=1)) # many 
        if all(  np.isclose(self.rrvv[idx, 0:3], r, 1e-8)  ): 
            return idx
        else: 
            return None

    def at(self, float_index): # param
        idx = int(float_index)
        if idx == self.rrvv.shape[0] - 1: 
            return self.rrvv[idx]
        
        t = float_index - idx
        idx1 = idx + 1
        rv0 = self.rrvv[idx]
        rv1 = self.rrvv[idx1]
        return rv0 + t*(rv1 - rv0)
    
    def find_origin(self, primary_traj): 
        rv0 = self.rrvv[0, 0:3]
        return primary_traj.find(rv0)


def obstacle_is_fatal(obstacles): 
    for ob in obstacles: 
        if not ob.startswith('_'): 
            return True
    return False
            
#%%

class StopPrim: 
    def __init__(self, geom, invisible_wall_x = 5.5): 
        self.geom = geom
        self.invisible_wall_x = invisible_wall_x
        #invisible_wall_x = self.geom.r_dict[target][0]+0.2
        
    def __call__(self, traj, rv_old, rv_new): 
        if rv_new[0] > self.invisible_wall_x and rv_new[1] < 1.2:
            traj.obstacle = "invisible_wall"
            return True

        if self.geom.check_chamb_intersect('prim', rv_old[0:3], rv_new[0:3]):
            traj.obstacle = "chamb_prim"
            return True

        plts_flag, plts_name = self.geom.check_plates_intersect(rv_old[0:3], rv_new[0:3])
        if plts_flag:
            traj.obstacle = plts_name
            return True

        if self.geom.check_fw_intersect(rv_old[0:3], rv_new[0:3]): 
            traj.obstacle = "first_wall"
            return True  # stop primary trajectory calculation        
        
        return False
    

class StopSec: 
    def __init__(self, geom, aim, invisible_wall_x = None): 
        self.aim = aim
        self.geom = geom
        self.r_aim = aim.r
        if invisible_wall_x is None: 
            self.invisible_wall_x = self.r_aim[0] + 0.1 # invisible_wall_x  # invisible_wall_x = self.geom.r_dict[target][0]+0.2
        else: 
            self.invisible_wall_x = invisible_wall_x
        
        self.stop_plane_n = aim.normal

    def check_intersect(self, rv_old, rv_new): 
        if rv_new[0] > self.invisible_wall_x: 
            return "invisible_wall"
            
        if self.geom.check_chamb_intersect('sec', rv_old[0:3], rv_new[0:3]):
            return "chamb"

        plts_flag, plts_name = self.geom.check_plates_intersect(rv_old[0:3], rv_new[0:3])
        if plts_flag:            
            return plts_name
        
        return None
        
        
    def __call__(self, traj, rv_old, rv_new):
        r_aim = self.r_aim
        eps_xy = 10e-3
        eps_z = 10e-3

        obstacle = self.check_intersect(rv_old, rv_new)
        if obstacle is not None: 
            traj.obstacle = obstacle
            return True
        
        # optimization: skip check intersect
        if (rv_new[0] > 2.5 + 1) and (rv_new[1] < 1.5): 
            return False # continue

        # intersection with the stop plane:
        #r_intersect = gf.plane_segment_intersect(self.stop_plane_n, r_aim, rv_old[:3], rv_new[:3])
        r_intersect = self.aim.intersect_segm(rv_old[:3], rv_new[:3])
        if r_intersect is not None: 
            delta_r = r_intersect - r_aim
            
            # check XY plane:            
            if vNorm(delta_r[:2]) <= eps_xy:
                traj.success = 'AimXY'  #  
                traj.r_intersect = r_intersect                   
                return True
            
            # check XZ plane:
            if (vNorm(delta_r[2]) <= eps_z):
                traj.success = 'AimZ' # traj.IsAimZ = True  # 
                traj.r_intersect = r_intersect
                return True
            
            traj.r_intersect = r_intersect
            traj.obstacle = 'aim_plane'
            traj.success = None
            return True
        
        return False

class OneStep: 
    def __init__(self): 
        pass

    def __call__(self, traj, rv_old, rv_new):
        return True
    
class Aim: 
    def __init__(self, r, normal, basis): 
        self.r = r
        self.normal = normal
        # basis 
        self.e_principal, self.e_lateral, self.e_onward = basis # ???
    
        self.zone_size = 0.02
    

    @classmethod
    def from_plates(cls, pl): 
        #r = pl.r        
        rect = pl.front_rect()
        r = gf.rect_center(rect)  # !!! sic!

        v, h = pl.front_basis(norm=True)
        normal = np.cross(v, h)
        basis = (v, h, normal)
        return Aim(r, normal, basis)

    @classmethod
    def from_analyzer(cls, geom): 
        r = geom.r_dict['an']
        an = geom.plates_dict['an']
        normal = an.slit_plane_n
        #r = an.r

        _v, _h = an.front_basis(norm=True)
        #_normal = np.cross(v, h)
        v = np.cross(normal, _v); v = np.cross(normal, v)
        h = np.cross(normal, _h); h = np.cross(normal, h)        
    
        basis = (-v, -h, normal)
        return Aim(r, normal, basis)

        # calculate normal to slit plane:

    def basis(self): 
        return self.e_principal, self.e_lateral, self.e_onward
    
    def plot(self): 
        v, h, a = self.basis()
        gf.plot_segm(self.r, self.r + v*0.05, color='r')
        gf.plot_segm(self.r, self.r + h*0.05, color='g')
        gf.plot_segm(self.r, self.r + a*0.05, color='black')
        
    def intersect_segm(self, r0, r1): 
        return gf.plane_segment_intersect(self.normal, self.r, r0, r1)

    def intersect_traj(self, traj): 
        for r0, r1 in traj.segments(): 
            r_int = self.intersect_segm(r0, r1) 
            if r_int is not None: 
                return r_int

        return None    
    
    
#tr.run(E, B, stop=StopPrim(geomT15))
#tr.plot()

#------------------------------------------------------------------------------

def mask_for_secondaries(rrvv, geom): 
    mask = np.sqrt((rrvv[:, 0] - geom.R)**2 + (rrvv[:, 1] / geom.elon)**2) <= geom.r_plasma
    return mask


def pass_fan(aim, E, B, geom, full_fan=False): 
    '''
    passing fan from initial point self.RV0
    '''
    # ********************************************************* #               
    dt = 0.2e-7 
    tr = Trajectory.from_injector(SI_e, SI_AEM*204.0, 240.0, geomT15.r_dict['r0'], geomT15.angles_dict['r0'][0], geomT15.angles_dict['r0'][1], dt)
    tr.run(E, B, stop=StopPrim(geom))
    
    if tr.obstacle in ['chamb_prim', 'A1', 'A2', 'B1', 'B2']: 
        return tr
    
    mask = mask_for_secondaries(tr.rrvv, geom)

    rrvv0_sec = tr.rrvv[mask]
    for rv0 in rrvv0_sec: 
        tr2 = Trajectory(tr.q*2.0, tr.m, tr.Ebeam, rv0, dt)
        tr2.run(E, B, stop=StopSec(geom, aim) )
        if tr2.r_intersect is not None: 
            tr2.rrvv[-1, 0:3] = tr2.r_intersect

        if tr2.success or full_fan:
            tr.fan.append(tr2)   
    
    return tr

#------------------------------------------------------------------------------

U_dict = {'A2': 14.0, 'B2': 0.0, 'A3': -38.4, 'B3': 0.3, 'A4': 0.1, 'an': 50.0}

B = Bfield(Binterp)
E = Efield(E_fast, U_dict, geomT15)
#E = Efield0()

#dt = 0.2e-7 
#tr = Trajectory.from_injector(SI_e, SI_AEM*204.0, 220.0, geomT15.r_dict['r0'], geomT15.angles_dict['r0'][0], 
#             geomT15.angles_dict['r0'][1], dt)

#analyzer = geomT15.plates_dict['an']
#analyzer.plot(plt.gca())
geomT15.plot(plt.gca())
  
def pass2target(tr, aim, E, B, geom):
    #set basis
    eps = 0.01  # cm
    v, h, a = aim.basis()
    ddrr = [tr2.rrvv[-1, 0:3] - aim.r for tr2 in tr.fan]
 
    dr_projections  = [dr.dot(v) for dr in ddrr if abs(dr.dot(a)) < eps]
    #dr_projections = [dr.dot(v) for dr in ddrr]
    i0, i1, t = misc.find_fork(dr_projections)
    
    if t is not None: 
        dt = tr.dt*t
        rv0 = tr.fan[i0].rrvv[0]

        tr_ = Trajectory(tr.q, tr.m, tr.Ebeam, rv0, dt) 
        tr_.run(E, B, OneStep()) # ??? 
        i2insert = tr.find(tr_.rrvv[0, :3])
        tr.rrvv = np.insert(tr.rrvv, i2insert, tr_.rrvv[-1], axis=0)

        rv0 = tr_.rrvv[-1]
        tr2 = Trajectory(tr.q*2.0, tr.m, tr.Ebeam, rv0, dt) 
        tr2.run(E, B, StopSec(geom, aim))
        tr.secondary = tr2
    else: 
        print('Oops')

class Optimizer: # ???
    def __init__(self, uname, aim, geom, init_traj): 
        self.aim = aim
        self.geom = geom
        self.uname = uname # 'A3'
        self.udomain = np.linspace(-40.0, 40.0, 41)
        self.init_traj = init_traj
        self.temp_traj = None
        
    #def setU(self, u): 
    #    self.E.U[self.uname] = u

    def calc_discrep(self, E, B, u): 
        stop = StopSec(self.geom, self.aim)
        E.U[self.uname] = u
        tr = deepcopy(self.init_traj)
        tr.run(E, B, stop=stop)
        v, h, along = self.aim.basis()
        d = v.dot(tr.rrvv[-1, 0:3] - self.aim.r)
        self.temp_traj = tr
        return d # if vNorm(along.dot(tr.rrvv[-1] - self.aim.r) ) < eps
        
    def optimize(self, E, B, geom):
        uu = self.udomain
        #dd = np.zeros_like(uu)
        dd = [self.calc_discrep(E, B, u) for u in uu] 

        i0, i1, t = misc.find_fork(dd, threshold=self.aim.zone_size) 

        if i0 is not None: 
            u_ok =  uu[i0] + (uu[i1]-uu[i0])*t 
            E.U[self.uname] = u_ok
            _ = self.calc_discrep(E, B, u_ok) # tr._pass_sec(RV0, optimizer.r_target, E, B, geom, tmax,  eps_xy, eps_z, optimizer.stop_plane_n, break_at_intersection=True)
            return u_ok, self.temp_traj
        else: 
            return None, None


plt.figure(1)
r_aim = geomT15.r_dict["aim"]

aim = Aim.from_plates( geomT15.plates_dict['A3'] ) 
aim.plot()

tr = pass_fan(aim, E, B, geomT15, True)    

pass2target(tr, aim, E, B, geomT15)
aim2 = Aim.from_plates( geomT15.plates_dict['A4'] )
tr.secondary.run(E, B, StopSec(geomT15, aim2, invisible_wall_x = 5.0))


tr.plot()
#for tr2 in tr.fan: 
#    tr2.plot()
tr.secondary.plot()

#op = Optimizer('A3', aim2, geomT15, tr.trim(tr.secondary.rrvv[0]))
op = Optimizer('A3', aim2, geomT15, tr.secondary.trim(tr.secondary.rrvv[0, 0:3]) )
uA3, _tr = op.optimize(E, B, geomT15)
#_tr.plot()

aim3 = Aim.from_analyzer( geomT15 )
op = Optimizer('A4', aim3, geomT15, _tr )
uA4, _tr = op.optimize(E, B, geomT15)
_tr.plot()


#pl = geomT15.plates_dict['A3']
#print(pl.r)
#print(geomT15.r_dict[pl.name])


#%%
if False: 
    geomT15.plot(plt.gca())
    tr.plot()
    tr.secondary.plot()

#%%
v, h, a = aim.basis()
ddrr = [tr2.rrvv[-1, 0:3] - aim.r for tr2 in tr.fan]
 
#dr_projections  = [dr.dot(v) for dr in ddrr if abs(dr.dot(a)) < eps]
dr_projections = [dr.dot(v) for dr in ddrr]
#plt.plot(dr_projections)

#%%

tr2 = tr.fan[1] 

r_int = aim.intersect_traj(tr2)
print(r_int)
#for 


