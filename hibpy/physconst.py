# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:12:46 2022

@author: reonid
"""

import numpy as np

PI = 3.1415926535897932384626433832795

SI_AEM     = 1.6604e-27       #     {kg}
SI_e       = 1.6021e-19       #     {C}
SI_Me      = 9.1091e-31       #     {kg}      // mass of electron
SI_Mp      = 1.6725e-27       #     {kg}      // mass of proton
SI_c       = 2.9979e8         #     {m/sec}   // velocity of light
SI_1eV     = SI_e             #     {J}
SI_1keV    = SI_1eV*1000.0    #     {J}
SI_h       = 6.626068e-34     #     {J*sec}   // Planck constant h
SI_hbar    = 1.054571e-34     #     {J*sec}   // Planck constant h-bar = h/2pi
SI_k       = 1.3807e-23       #     {J*K-1}

SI_epsilon0 = 8.8541878e-12   #     {F/m}
SI_mu0      = 4*PI*1e-7       #     {H/m}
SI_Avogadro = 6.022136736     #     {1/mol}

SI_Cu_resistivity = 0.0175e-6 #     {Ohm*m}   //0.0172..0.18
SI_Cu_conductivity = 1/SI_Cu_resistivity

SI_1statV  = 299.792458       #     {V}
CGS_1V     = 1/SI_1statV      #     {statV}

CGS_AEM    = SI_AEM*1E3       #     {g}
CGS_c      = SI_c*100         #     {cm/sec}
CGS_e      = 4.80320427e-10   #     {statcoulomb = erg^0.5*cm^0.5}

A_Tl       = 204.37
A_Cs       = 132.9054
A_K        = 39.098

SI_M_p     = SI_Mp            #     {kg}
SI_M_D     = 2.0*SI_Mp        #     {kg}

SI_M_Tl    = A_Tl*SI_AEM      #     {kg}
SI_M_Cs    = A_Cs*SI_AEM      #     {kg}
SI_M_K     = A_K*SI_AEM       #     {kg}

CGS_M_Tl   = SI_M_Tl*1e3      #     {g}
CGS_M_Cs   = SI_M_Cs*1e3      #     {g}
CGS_M_K    = SI_M_K*1e3       #     {g}


SI_CELSIUS_0 = 273.15         #     {K}


#%%

def LnLambda():
    return 15.0   # 10..15

#def qLim(R_cm, a_cm, Bz_kOe, Ipl_kA): 
#    return 5.0*Bz_kOe*(a_cm**2)/(R_cm*Ipl_kA)

def qLim(R_m, a_m, Bz_T, Ipl_kA): 
    #if Ipl_kA < 0.001: return 0.0
    
    R_cm = R_m*100.0
    a_cm = a_m*100.0
    Bz_kOe = Bz_T*10.0
    #return 5.0*Bz_kOe*(a_cm**2)/(R_cm*Ipl_kA)
    return 5.0*Bz_kOe*(a_cm*a_cm)/(R_cm*Ipl_kA)

def qLim_astra(R_m, a_m, Bz_T, Ipl_kA, kappa=1.0, delta=0.0): 
    # qLim in the toroidal geometry
    # kappa - elongation, delta - triangularity
    Ipl_MA = Ipl_kA*1e-3
    q0 = 5.0*a_m*a_m*Bz_T/R_m/Ipl_MA
    k2 = kappa*kappa
    d2 = delta*delta
    d3 = d2*delta
    aR = a_m/R_m
    aR2 = aR*aR
    term1 = 0.5*(1.0 + k2*(1.0 + 2.0*d2 - 1.2*d3))
    term2 = (1.17 - 0.65*aR)/(1.0 - aR2)**2
    return q0*term1*term2


'''
def tokamak_q(R_m, r_m, Bz_T, jj_kA_m2, rr_m): 
    R_cm = R_m*100.0
    r_cm = r_m*100.0
    Bz_kOe = Bz_T*10.0
    
    jj_kA_cm2 = jj_kA_m2*10000.0
    rr_cm = rr_m*100.0
    I_kA = np.trapz(rr_cm, jj_kA_cm2)
    
    if isinstance(r_cm, np.ndarray): 
        result = np.zeros(r_cm)
        5.0*Bz_kOe*(r_cm*r_cm)/(R_cm*I_kA)
        return result 
    else: 
        return 5.0*Bz_kOe*(r_cm*r_cm)/(R_cm*I_kA)
'''


def SI_qLim(R_m, a_m, Bz_T, Ipl_A): 
    #if Ipl_A < 1.0: return 0.0
    
    R_cm = R_m*100.0
    a_cm = a_m*100.0
    Bz_kOe = Bz_T*10.0
    Ipl_kA = Ipl_A*0.001
    return 5.0*Bz_kOe*(a_cm**2)/(R_cm*Ipl_kA)

def gyro_radius(m_kg, Z, Bz_T, T_keV): # m
    E_J = T_keV*SI_1keV
    v_m_s = np.sqrt(2.0*E_J/m_kg)
    q_C = np.abs(Z*SI_e)
    return m_kg*v_m_s/q_C/Bz_T

def _gyro_radius(particle, Bz_T, T_keV): # m
    # for test
    kk = {'e': 1.066e-4, 'p': 4.570e-3, 'D': 6.461e-3}
    k = kk[particle]
    return k*np.sqrt(T_keV)/Bz_T


#%%

def alfven_velocity(ne_19, Bz_T, ion): # {m/s}
    if ne_19 == 0: return 0.0

    CGS_ne = ne_19 * 1e13   # {cm^-3}
    CGS_Bz = 10000.0*Bz_T   # {Gs}
    #CGS_Valf = 2.18E11 / np.sqrt(ion * CGS_ne) * CGS_Bz; # {cm/s}
    CGS_Valf = CGS_Bz / np.sqrt(ion * CGS_AEM * 4.0*PI * CGS_ne)

    return CGS_Valf * 0.01 #  {m/s}

def alfven_freq(ne_19, Bz_T, R_m, iota, ion, M, N):  #  {Hz}
    SI_Valf = alfven_velocity(ne_19, Bz_T, ion)
    SI_omega = 1/R_m * np.abs(N - iota*M) * SI_Valf
    return SI_omega/2.0/PI


def velocity(m_kg, E_keV): # m/s
    E_J = E_keV*SI_1keV
    return np.sqrt(2.0*E_J/m_kg)


#%%

def spitzer_resistivity_par(Te_keV, Zeff, LnLambda): #  Ohm*m
    return spitzer_resistivity_unmagnetized(Te_keV, Zeff, LnLambda)

def spitzer_resistivity_unmagnetized(Te_keV, Zeff, LnLambda): #  Ohm*m 
    Te_eV = Te_keV*1000.0; 
    return 5.2e-5*Zeff*LnLambda/(Te_eV**1.5)

def spitzer_resistivity_unmagnetized_Z1(Te_keV, LnLambda): #  Ohm*m 
    return 1.65e-9*LnLambda/(Te_keV**1.5)


#%%

def KirnevaOhmicApproxTe(rho, ne_19, Bz_T, Ipl_kA): # {keV}
    """
    T-10 Ohmic Te scaling
    """    
    rbig_m = 1.5     # {m}
    rsmall_m = 0.3   # {m}
    rho = np.abs(rho)
    q = qLim(rbig_m, rsmall_m, Bz_T, Ipl_kA)
    Ipl_norm24_kA = 720.0/q         # [kA] Ipl normalized on Bz = 2.4
    #ne_k := ne_19*0.75           #  normalized on camera 40 cm

    A = 1400.0*(1 - np.exp(   -(Ipl_norm24_kA-48.0)/25.0    ))     
    if A < 0: A = 0
    B =  0.3*(1.0 - np.exp(   -(Ipl_norm24_kA-75.0)/25.0    ))     
    # if B < 0 then B := 0;

    # Central Te
    Te0_eV = A * (ne_19 + 0.5)**(-B) # {eV}
    Te0_keV = Te0_eV*0.001 # {keV}

    alpha = (Ipl_norm24_kA**0.435)/5.0; 
    beta = 2.5 - (Ipl_norm24_kA - 112.5)**2/50000.0;

    radial_coef = 1 - rho**alpha
    radial_coef = radial_coef**beta

    Te_keV = Te0_keV*radial_coef
    return Te_keV
    

if __name__ == '__main__':
    print('qLim = ', SI_qLim(1.5, 0.30, 2.2, 250e3))
    print('qLim = ', qLim(1.5, 0.30, 2.2, 250.0))
    print('qLim_astra = ', qLim_astra(1.5, 0.30, 2.2, 250.0))
    
    import matplotlib.pyplot as plt
    
    xx = np.linspace(-1.0, 1.0, 100)
    yy = KirnevaOhmicApproxTe(xx, ne_19=2.0, Bz_T=2.2, Ipl_kA=80.0)
    plt.plot(xx, yy)
    
    yy = KirnevaOhmicApproxTe(xx, ne_19=2.0, Bz_T=2.2, Ipl_kA=280.0)
    plt.plot(xx, yy)

