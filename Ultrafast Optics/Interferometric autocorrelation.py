# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:53:42 2021

@author: shubh
"""
import numpy as np
import math
from scipy import integrate
import cmath
import matplotlib.pyplot as plt


#defining variables
t=np.arange(-10**(-12),10**(-12),10**(-16))
width = 100*10**(-15)
c=3*10**8
wavelength = 800*10**(-9)
central_freq=2*np.pi*c/wavelength
metric = width/(np.sqrt(2*np.log(2)))
chirp_metric=50*10**(-15)


#defining functions
def env(i):
    return np.exp(-(i/metric)**2)

def multiple(j,n):
    return np.exp(-complex(0,1)*n*central_freq*j)

def cphase(c):
    return -(c/chirp_metric)**2
#    return 0

def integrand1(k):
    func1=lambda x: ((env(x-k))**4)+((env(x))**4)+4*((env(x-k))**2)*((env(x))**2)
    return func1

def integrand2(l):
    func2=lambda x: (env(x-l) * env(x) * (((env(x-l))**2)+((env(x))**2)) * np.exp(complex(0,1)*((cphase(x-l))-(cphase(x)))))
    return func2

def integrand3(m):
    func3 = lambda x: (((env(x-m))**2)*((env(x))**2)* np.exp(2*complex(0,1)*((cphase(x-m))-(cphase(x)))))
    return func3

def function (j):
    func = lambda x : np.abs(np.exp(-(x/metric)**2))**2*np.abs(np.exp(-((x-j)/metric)**2))**2
    return func


yaxis = np.array([integrate.quadrature(function(i),-5*width,5*width) for i in t])

#performing the integration
A0 = np.array([integrate.quadrature(integrand1(i),-5*width,5*width,tol=10**-85) for i in t],dtype="complex_")  
A1 = np.array([integrate.quadrature(integrand2(j),-5*width,5*width,tol=10**-85) for j in t],dtype="complex_")  
A2 = np.array([integrate.quadrature(integrand3(k),-5*width,5*width,tol=10**-85) for k in t],dtype="complex_")    
   
sec_term=[]
for i, j in zip(t, A1[:,0]):
    sec_term.append((multiple(i, 1)*j))
    
third_term=[]
for i, j in zip(t, A2[:,0]):
    third_term.append((j*multiple(i, 2)))
    
sec_term = np.array((sec_term)).real
third_term = np.array((third_term)).real
#
final = (4*sec_term + 2*third_term + A0[:,0]).real
ia=np.abs(4*sec_term)

#import pickle
#with open("final.pkl", "rb") as f:
#    final = pickle.load(f)
##final = (((4*sec_term + 2*third_term)))
plt.clf()
plt.xlabel('delay(s)',fontsize=15)
plt.ylabel('Interferometric autocorrelation(a.u)',fontsize=15)
plt.plot(t,(final)/np.max(final), color='royalblue')
#plt.plot(t,yaxis/np.max(yaxis), color='indigo')
plt.legend(['Interferometric autocorrelation with chirp'],loc='upper right',prop={"size":15})
plt.xlim(-3e-13,3e-13)   
#plt.plot(t,(A0[:,0])/np.max(A0[:,0]))    
    