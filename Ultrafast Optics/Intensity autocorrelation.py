# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 21:30:02 2021

@author: shubh
"""

import numpy as np
import math
from scipy import integrate
import cmath
import matplotlib.pyplot as plt

#defining variables
t=np.arange(-10**(-12),10**(-12),10**(-14))
width = 200*10**(-15)
wavelength = 800*10**(-9)
metric = width/(np.sqrt(2*np.log(2)))
chirp_metric=100*10**(-15)
c=3*10**8
print(metric)

def int_env(i):
    return np.abs(np.exp(-(i/metric)**2))**2

def int_chirp_env1(j):
#    return np.abs(np.exp(-((j/chirp_metric)**2)))**2
    return np.abs(np.exp(-(complex(0,1)*(j/chirp_metric)**2)))**2

def int_chirp_env2(j):
#    return np.abs(np.exp(-((j/chirp_metric)**2)))**2
    return 1

def function1(k):
    func=lambda x: (int_env(x)*int_env(x-k)*int_chirp_env1(x)*int_chirp_env1(x-k))
    return func
def function2(l):
    func=lambda x: (int_env(x)*int_env(x-l)*int_chirp_env2(x)*int_chirp_env2(x-l))
    return func

xaxis = np.linspace(-10**(-15),10**(-15),2000)
yaxis1 = np.array([integrate.quadrature(function1(i),-5*width,5*width) for i in t],dtype="complex_")
yaxis2 = np.array([integrate.quadrature(function2(i),-5*width,5*width) for i in t],dtype="complex_")

plt.figure("Intensity vs f")
plt.clf()
plt.xlabel('delay (sec)',fontsize=15)
plt.ylabel('Power (a.u.)',fontsize=15)
plt.plot(t,(yaxis1[:,1]/np.max(yaxis1[:,1])),color='royalblue', label='Intensity autocorrelation without chirp (tau=200fs)')
plt.plot(t,(yaxis2[:,1]/np.max(yaxis2[:,1])),color='red', label='Intensity autocorrelation with chirp of 100fs (tau=200fs)')
plt.legend(loc='upper right',prop={"size":15})
#plt.legend(['Intensity autocorrelation without chirp (tau=200fs)','Intensity autocorrelation with chirp of 100fs (tau=200fs)'],
#            loc='upper right',prop={"size":15})

def FWHM(X, Y):
    deltax = X[1] - X[0]
    half_max = np.max(Y) / 2.
    l = np.where(Y > half_max, 1, 0)

    return np.sum(l) * deltax

o=FWHM(t,yaxis2/np.max(yaxis2))
print(o)