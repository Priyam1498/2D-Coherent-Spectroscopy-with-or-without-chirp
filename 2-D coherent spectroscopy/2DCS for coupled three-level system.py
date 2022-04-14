# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:15:47 2022

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
Hz2meV = 4.13*10**-12 #meV
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
c = 0 #chirp
#T = 5*np.pi*sigma
T = np.pi*sigma
#T = 0
#
gamma01 = gamma10 = 0.2*b
gamma11 = gamma22 = gamma00 = gamma21 = gamma12 = 0.1*b
gamma02 = gamma20 = 0.2*b

w0 = 3*meV2Hz*2*np.pi
ex_energy0 = w0*Hz2meV/(2*np.pi)
w10 = w0-(b/(2))
Energy10 = (b*Hz2meV/2)/(2*np.pi)

#w10 = (wL/2)-(b)
w01 = -w10
w20 = w0+(b/(2))
#w20 = 1*meV2Hz*2*np.pi
Energy20 = (b*Hz2meV/2)/(2*np.pi)
wL = (w10+w20)/2
w02 = -w20
w21 = w20-w10
w12 = -w21
w00 = w11 = w22 = 0

omega01 = w01 - (1j)*gamma01
omega10 = w10 - (1j)*gamma10
omega20 = w20 - (1j)*gamma20
omega02 = w02 - (1j)*gamma02
omega00 = w00 - (1j)*gamma00
omega11 = w11 - (1j)*gamma11
omega22 = w22 - (1j)*gamma22
omega21 = w21 - (1j)*gamma21
omega12 = w12 - (1j)*gamma12
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
#multiple1 = np.zeros((n,n),dtype = complex)
spectrum = np.zeros((n,n),dtype = complex )
n1=-1
n2=n3=-n1


#For first feynmann diagram
ohi1 = omega00
ofg1 = omega01
ojk1 = omega20
spectrum1 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum1[j,i] = (gaussianc(wt[i],wtau[j],ohi1)*hp(wt[i],wtau[j],ojk1,ofg1))*errpar(wt[i],wtau[j],n2,n3,ohi1)         
                
#For Second Feynmann diagram
ohi2 = omega21
ofg2 = omega01
ojk2 = omega20  
spectrum2 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum2[j,i] = (gaussianc(wt[i],wtau[j],ohi2)*hp(wt[i],wtau[j],ojk2,ofg2))*errpar(wt[i],wtau[j],n2,n3,ohi2) 
                
#For Third Feynmann diagram
ohi3 = omega12
ofg3 = omega02
ojk3 = omega10  
spectrum3 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum3[j,i] = (gaussianc(wt[i],wtau[j],ohi3)*hp(wt[i],wtau[j],ojk3,ofg3))*errpar(wt[i],wtau[j],n2,n3,ohi3) 
                
#For Fourth Feynmann Diagram
ohi4 = omega00
ofg4 = omega02
ojk4 = omega10  
spectrum4 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum4[j,i] = (gaussianc(wt[i],wtau[j],ohi4)*hp(wt[i],wtau[j],ojk4,ofg4))*errpar(wt[i],wtau[j],n2,n3,ohi4) 

#For Fifth Feynmann diagram
ohi5 = omega11
ofg5 = omega01
ojk5 = omega10  
spectrum5 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum5[j,i] = (gaussianc(wt[i],wtau[j],ohi5)*hp(wt[i],wtau[j],ojk5,ofg5))*errpar(wt[i],wtau[j],n2,n3,ohi5) 

#For Sixth Feynmann diagram
ohi6 = omega00
ofg6 = omega01
ojk6 = omega10  
spectrum6 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum6[j,i] = (gaussianc(wt[i],wtau[j],ohi6)*hp(wt[i],wtau[j],ojk6,ofg6))*errpar(wt[i],wtau[j],n2,n3,ohi6)  

#For Seventh Feynmann diagram
ohi7 = omega22
ofg7 = omega02
ojk7 = omega20  
spectrum7 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum7[j,i] = (gaussianc(wt[i],wtau[j],ohi7)*hp(wt[i],wtau[j],ojk7,ofg7))*errpar(wt[i],wtau[j],n2,n3,ohi7) 

#For Eighth Feynmann diagram
ohi8 = omega00
ofg8 = omega02
ojk8 = omega20  
spectrum8 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum8[j,i] = (gaussianc(wt[i],wtau[j],ohi8)*hp(wt[i],wtau[j],ojk8,ofg8))*errpar(wt[i],wtau[j],n2,n3,ohi8)


spectrum = spectrum1+spectrum2+spectrum3+spectrum4+spectrum5+spectrum6+spectrum7+spectrum8

#%% Plotting in freq domain
plt.figure('Spectra in freq domain')
plt.clf()
plt.title (r'for T=$\pi\sigma$')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.pcolormesh(ex_energy,em_energy,np.abs(spectrum)/np.amax(np.abs(spectrum)),shading='auto', cmap='jet')
plt.colorbar()
plt.show()

#%%Forming arrays 
#multiple1 = np.zeros((n,n),dtype = complex)
spectrum0 = np.zeros((n,n),dtype = complex )
n1=-1
n2=n3=-n1


#For first feynmann diagram
ohi1 = omega00
ofg1 = omega01
ojk1 = omega20
spectrum10 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum10[j,i] = hp(wt[i],wtau[j],ojk1,ofg1)*(np.exp(-1j*ohi1*T)) 
    
#For Second Feynmann diagram
ohi2 = omega21
ofg2 = omega01
ojk2 = omega20  
spectrum20 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum20[j,i] =hp(wt[i],wtau[j],ojk2,ofg2)*(np.exp(-1j*ohi2*T))
                
#For Third Feynmann diagram
ohi3 = omega12
ofg3 = omega02
ojk3 = omega10  
spectrum30 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum30[j,i] = hp(wt[i],wtau[j],ojk3,ofg3)*(np.exp(-1j*ohi3*T)) 
                
#For Fourth Feynmann Diagram
ohi4 = omega00
ofg4 = omega02
ojk4 = omega10  
spectrum40 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum40[j,i] = hp(wt[i],wtau[j],ojk4,ofg4)*(np.exp(-1j*ohi4*T))

#For Fifth Feynmann diagram
ohi5 = omega11
ofg5 = omega01
ojk5 = omega10  
spectrum50 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum50[j,i] = hp(wt[i],wtau[j],ojk5,ofg5)*(np.exp(-1j*ohi5*T))

#For Sixth Feynmann diagram
ohi6 = omega00
ofg6 = omega01
ojk6 = omega10  
spectrum60 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum60[j,i] = hp(wt[i],wtau[j],ojk6,ofg6)*(np.exp(-1j*ohi6*T))
#For Seventh Feynmann diagram
ohi7 = omega22
ofg7 = omega02
ojk7 = omega20  
spectrum70 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum70[j,i] = hp(wt[i],wtau[j],ojk7,ofg7)*(np.exp(-1j*ohi7*T)) 

#For Eighth Feynmann diagram
ohi8 = omega00
ofg8 = omega02
ojk8 = omega20  
spectrum80 = np.zeros((n,n),dtype = complex )
for i in range(len(wt)):
            for j in range(len(wtau)):
                spectrum80[j,i] = hp(wt[i],wtau[j],ojk8,ofg8)*(np.exp(-1j*ohi8*T))


spectrum0 = spectrum10+spectrum20+spectrum30+spectrum40+spectrum50+spectrum60+spectrum70+spectrum80

#%% Plotting in freq domain
plt.figure('Spectra0 in freq domain')
plt.clf()
plt.title (r'for T=$\pi\sigma$')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.pcolormesh(ex_energy,em_energy,np.abs(spectrum0)/np.amax(np.abs(spectrum0)),shading='auto', cmap='jet')
plt.colorbar()
plt.show()

#%%
qspectrum = np.divide((spectrum),(spectrum0),dtype =complex)
def colorize(z):
    z=np.flipud(z)
    r = np.abs(z)/np.amax(np.abs(z))
    arg = r*np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
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
plt.title (r'for T=$\pi\sigma$')
plt.xlabel('Emission energy (meV)',fontsize=12)
plt.ylabel('Excitation Energy (meV)',fontsize=12)
plt.imshow(img/np.amax(img),extent=[-En,En,-En,En])

plt.contour(ex_energy,em_energy,abs(z)/np.amax(abs(z)),levels=4,colors='white')

plt.show()



