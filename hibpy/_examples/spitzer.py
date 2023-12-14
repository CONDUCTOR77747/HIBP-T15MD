# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:32:50 2022

@author: reonid
"""

import sys
hibplib_path = 'd:\\reonid\\myPython\\reonid-packages'
if hibplib_path not in sys.path: sys.path.append(hibplib_path)

import numpy as np
from hibplib.xysig import XYSignal
from hibplib.fit.bessfit import bessfit2
from hibplib.fit.hollow import hollow_func, hokusai_func


mu0 = 4.0*np.pi*1e-7  #;      {H/m}
8.85419e-12
ε0 = 8.85419e-12
μ0 = mu0
π = np.pi

def Spitzer_Resistivity_SingleCharged(Te_keV, LnLambda): # Ohm*m
    return 1.65e-9*LnLambda/(Te_keV**1.5)

def Spitzer_Conductivity_SingleCharged(Te_keV, LnLambda):
    #Zeff = 1.0
    #return 1.9e4*Te_keV**1.5/LnLambda/Zeff
    return 1.0/Spitzer_Resistivity_SingleCharged(Te_keV, LnLambda)

LnLambda = 12 # 10..15
Te_keV = 0.6 # *0.9
R_m = 1.5
a_m = 0.22
c = 2.9979e8 # 

L = 2.0*π*R_m # m
S = π*a_m**2

l_small = 0.5
L_big = 0.5*l_small*μ0*R_m

R_Ohm = Spitzer_Resistivity_SingleCharged(Te_keV, LnLambda)*L/S

sigma = Spitzer_Conductivity_SingleCharged(Te_keV, LnLambda)
tau_skin = (π*μ0*sigma*a_m**2) / 16

tau_skin = L_big/R_Ohm

tau_skin = (a_m**2)*μ0*sigma

print(R_Ohm, 'Ohm')
print(tau_skin*1000.0, 'ms')


#V = 0.9 
#a = np.sqrt(V/(2.0*π*π*R_m))
#print('a=', a)

xx = np.linspace(0.0, 1.0, 1000)
yy = hollow_func(xx, 1.335033, 1)

#yy = hokusai_func(xx, 1.0, 2)

norm_prof = XYSignal((xx, yy))

Te = norm_prof * Te_keV
Te.plot()

_R_Ohm = 0.0 
S = 0.0
for i in range(1000):
    L = 2.0*π*R_m
    r = i*a_m/1000.0
    dr = a_m/1000.0
    dS = 2.0*π*r * dr
    Te = Te_keV*hollow_func(i/1000.0, 1.335033, 1)
    #Te = Te_keV*hokusai_func(i/1000.0, 1.0, 1.5)

    ρ = Spitzer_Resistivity_SingleCharged(Te, LnLambda)
    _R_Ohm += 1.0/ (ρ*L/dS)
    S += dS

R_Ohm = 1/_R_Ohm
print(R_Ohm, 'Ohm')

sigma = 1.0/R_Ohm*L/S
τ_skin = (L**2)*μ0*sigma



τ_skin = L_big/R_Ohm
print('τ_skin = ', τ_skin*1000, ' ms')

#Δδ
Δ = a_m
freq = (503.0/Δ)**2/sigma

freq = 10.0 # Hz
omega = freq*2*π
rho = 1/sigma
Δ = c*np.sqrt(2*ε0*rho/omega)
print('Δ=', Δ*100, ' cm')
