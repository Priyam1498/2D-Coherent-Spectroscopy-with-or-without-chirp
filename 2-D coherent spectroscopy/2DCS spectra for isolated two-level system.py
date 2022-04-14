# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:57:41 2022

@author: shubh
"""

#%% Loading packages
import numpy as np
import math
import mpmath
import matplotlib.pyplot as plt
import cmath
from scipy import special
from colorsys import hls_to_rgb

#%% Defining variables
meV2Hz = 2.418*10**11 #Hz
En = 8
npts = 2**8
dEn = (2*En)/(npts)
ex_energy = np.arange(-En,En,dEn)
em_energy = ex_energy
wt = meV2Hz*2*np.pi * (ex_energy)
wtau = wt
n = len(ex_energy)
dt = 1/(wt[n-1]-wt[0])
t = np.array([dt*i for i in range(n)])
tau = t

sigma = (250*10**(-15))
b = (1)/sigma
c = 10  #chirp
#T = 4*sigma
T = 2*4*np.pi*sigma
#T = 0
gamma01 = gamma10 = 0.2*b
gamma11 = gamma00 = 0.1*b
wL = 1.5*meV2Hz*2*np.pi
w01 = -wL
w10 = wL
w00 = w11 = 0
omega01 = w01 - (1j)*gamma01
omega10 = w10 - (1j)*gamma10
omega00 = w00 - (1j)*gamma00
omega11 = w11 - (1j)*gamma11
n1=-1
n2=n3=-n1

#%% Defining Functions


def sigmac(n):
    sigmac = (np.sqrt(1-(1j*n*c)))*(1/b)
    return sigmac

def gaussian1c(x,n):
    gaussian1c = np.exp(-(((sigmac(n)*(x-(n*wL)))**2)/2))
    return gaussian1c

def gaussian2c(y,n,ohi):
    gaussian2c = np.exp(-(1/2)*((sigmac(n)*(y+(n*wL)-ohi))**2))
    return gaussian2c

def gaussian3c(z,n,ohi):
    gaussian3c = np.exp(-(1/2)*((sigmac(n)*(z-(n*wL)-ohi))**2))
    return gaussian3c

def gaussian4c(i):
    gaussian4c = np.exp(-(1j*c*((i-wL)**2)/(2*(b**2))))
    return gaussian4c

def gaussianc(i,j,ohi):                            #Fourier transform gaussian 
    gaussianc = np.exp(-1j*ohi*T)*gaussian1c(j,n1)*gaussian2c(j,n2,ohi)*gaussian3c(i,n3,ohi)*gaussian4c(i)
    return gaussianc

def hp(p,q,ojk,ofg):                         #Fourier-transformed Heaviside phase factor
    hp1 = (1/(p-ojk))*(1/(q-ofg))
    return hp1

def errpar1(r,s,n4,n5,ohi):
    errpar1 = (((sigmac(n5))**2)*(r-(n5*wL)-ohi))+(((sigmac(n4))**2)*(s+(n4*wL)-ohi))
    return errpar1

errparfact = 1/np.sqrt(2*((sigmac(n2))**2+(sigmac(n3))**2))
    

def errpararg(i,j,n4,n5,ohi):
    errpararg = (errparfact)*(T+ (1j*(errpar1(i,j,n4,n5,ohi))))
    return errpararg

def errpar(i,j,n4,n5,ohi):                                 #Error function
    errpar = (1/2)*(1+special.erf(errpararg(i,j,n4,n5,ohi)))
    return errpar
    
#%%Forming arrays 
spectrum = np.zeros((n,n),dtype = complex )


#For first feynmann diagram
ohi1 = omega00
ofg1 = omega01
ojk1 = omega10
spectrum1 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum1[j,i] = (gaussianc(wt[i],wtau[j],ohi1)*hp(wt[i],wtau[j],ojk1,ofg1))*errpar(wt[i],wtau[j],n2,n3,ohi1)


spectrum = 2*spectrum1
 

#%% Plotting in freq domain
plt.figure('Spectra for gaussian pulse in freq domain')
plt.clf()
plt.title (r'for T= $2\pi\sigma$')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.pcolormesh(ex_energy,em_energy,np.abs(spectrum)/np.amax(np.abs(spectrum)),shading='auto', cmap='jet')
plt.colorbar()
plt.show()

#%%Forming arrays 
spectrum3 = np.zeros((n,n),dtype = complex )
n1=-1
n2=n3=-n1


#For first feynmann diagram
ohi1 = omega00
ofg1 = omega01
ojk1 = omega10
spectrum2 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum2[j,i] = hp(wt[i],wtau[j],ojk1,ofg1)*np.exp(-1j*ohi1*T)


spectrum3 = 2*spectrum2

#%% Plotting in freq domain
plt.figure('Spectra for delta pulse in freq domain')
plt.clf()
plt.title (r'for T= 0')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.pcolormesh(ex_energy,em_energy,np.abs(spectrum3)/np.amax(np.abs(spectrum3)),shading='auto', cmap='jet')
plt.colorbar()
plt.show()


qspectrum = np.divide((spectrum),(spectrum3),dtype =complex)

#%%
def colorize(z):
    z=np.flipud(z)
    r = np.abs(z)/np.amax(np.abs(z))
    arg = np.angle(z) 
#   
    h = (arg + np.pi)  / (2*np.pi) + 0.5
    s = r
    l = 0.5 /(1.0 + s**1)
    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    c = c.swapaxes(0,1)
    return c

z = qspectrum
img = colorize(z)
plt.figure('Phase and contour')

plt.clf()
plt.title (r'for T= $2\pi\sigma$')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.imshow(img/np.amax(img),extent=[-En,En,-En,En])

plt.contour(ex_energy,em_energy,abs(z)/np.amax(abs(z)),levels=4,colors='white')

plt.show()
























