# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:24:55 2023

@author: Krohalev_OD
"""

#%% imports
import numpy as np
import hibpcalc.geomfunc as gf
import hibpcalc.fields as f
import hibpcalc.misc as m

#%%
class Trajectory():
    def __init__(self, q, m, Ebeam, r0, alpha, beta, U, dt=1e-7, V0=None):
        '''
        Parameters
        ----------
        q : float
            particle charge [Co]
        m : float
            particle mass [kg]
        Ebeam : float
            beam energy [keV]
        r0 : np.array
            initial point of the trajectory [m]
        alpha : float
            injection angle in XY plane [rad]
        beta : float
            injection angle in XZ plane [rad]
        U : dict
            dict of voltages in [kV] keys=[A1 B1 A2 B2 A3 B3 A4 an]
        dt : float, optional
            timestep for RK algorithm [s]. The default is 1e-7.
        V0 : np.array
            initial speed of particle. If None is given, will be calculated 
            from Ebeam

        Returns
        -------
        None.
        '''
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        self.alpha = alpha
        self.beta = beta
        self.U = U
        
        if V0 is None:
            Vabs = np.sqrt(2 * Ebeam * 1.602176634E-16 / m)
            V0 = gf.calc_vector(-Vabs, alpha, beta)
            
        self.RV = []
        self.RV.append(np.array([np.hstack((r0, V0))]))  # initial condition
        # list to contain trajectories of the whole fan:
        self.Fan = []
        # time step:
        self.dt = dt
        # flags
        self.IsAimXY = False
        self.IsAimZ = False
        self.fan_ok = False
        self.B_out_of_bounds = False
        self.log = []
        self.tags = []
    
    def print_log(self, s):
        '''
        Parameters
        ----------
        s : string
            any log message

        Returns
        -------
        None.
        '''
        self.log.append(s)
        print(s)
    
    def pass_self(self, E, B, geom, stop_criteria, aim_criteria=None, t_max=1e-5):
        '''
        Parameters
        ----------
        E : dict
            dict of FieldInterpolator objects with E field data for all plates and 
            analyzer
        B : FieldInterpolator
            Interpolator with B fiald data
        geom : Geometry
        stop_criteria : function, return boolean
            take trajectoryand return True if trajectory intersected geometry 
            or False if not
        aim_criteria : function, return tuple of boolean (IzAimXY, IzAimZ), 
                       optional
            take trajectory and return True if trajectory ended in aim zone
            or False if not
        t_max : float, optional
            The trajectory will be forced to stop if pass time exeeds t_max. 
            The default is 1e-5.

        Returns
        -------
        None.
        '''
        print('\n Passing primary trajectory')
        # reset intersection flags
        self.reset_flags()
        t = 0.
        RV_old = self.RV0  # initial position
        k = self.q / self.m
        self.tags.append(10)

        while t <= t_max:
            r = RV_old[0, :3]
            E_local = f.return_E(r, E, self.U, geom)
            B_local = B(r)
            if self.check_B_is_NaN(B_local):
                break

            RV_new = m.runge_kutt(k, RV_old, self.dt, E_local, B_local)
            
            if stop_criteria(self):
                break
            
            if not aim_criteria is None:
                self.IsAimXY, self.IsAimZ, intersection_point = aim_criteria(self)
           
            # save results
            self.RV.append(RV_new)
            self.tags.append(10)
            
            if self.IsAimXY and self.IsAimZ:
                self.cut(intersection_point)
            
            RV_old = RV_new
            t = t + self.dt

        else: 
            self.print_log('t <= tmax, t=%f' % t)
    
    def pass_fan(self, stop_criteria):
        pass
    
    def plot(self):
        pass