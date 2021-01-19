import numpy as np
import math

slope=2.2e5;

#universal constant
pi=3.14159
c=3.0e8*1e9/1e12                         #[nm/ps]
wavelength=1550                          #[nm]
f0=c/wavelength

#parameters of the resonator
L=628e-6                                #cavity length[m]
alpha=0.009                             #loss
delta0=-0.0045                          #initial cavity detuning

beta2=-7.0665e-2/1.5                    #GVD[ps^2/m]
beta3=-1.3908e-4                        #higher order dispertion[ps^2/m]
beta4=2.18e-6
beta5=-1.6249e-8
beta6=1.0405e-10
beta7=-4.4731e-13
beta8=9.1414e-16

gamma=1                                 #nonlinear coefficiency[1/W/m]
E_in=0.755                              #input power[W]
theta=0.009                             #transmission coefficiency
FSR=226e9                               #[Hz]
tr=1/FSR                                #roundtrip time[s]

#stepsize in z and tau
ht=tr/2
tmax=2*5650*ht
ntau=2**12                              #step numbers in tau
Tmax=2.21                              #window size(4.42ps)
dtau=2*Tmax/ntau

#tau and omega arrays
tau=np.arange(-ntau//2, ntau//2) * dtau  #temporal grid
omega = (pi / Tmax) * np.arange(-ntau/2, ntau/2) #frequency grid
freq = -omega / (2 * pi) + f0
lambdas = np.asarray([c / f for f in freq])
omega0 = 2 * pi * f0                   #central freq

#######Raman#######
# Defination of Raman response function
fR = 0.18
tau1 = 12.2e-3                         # ps
tau2 = 32e-3                           # ps

h = np.zeros(ntau)
for i in range(ntau // 2 + 1, int(ntau)):
    h[i] = fR * (tau1**2 + tau2**2) / (tau1*tau2**2) * math.exp(-tau[i] / tau2) * math.sin(tau[i]/tau1)