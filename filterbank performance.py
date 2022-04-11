
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

import deshima_sensitivity

### import csv files 
filter_data = np.genfromtxt('Directional_filter_data.csv',delimiter=',')
# filter_data = np.genfromtxt('Manifold_filter_data.csv',delimiter=',')






#### TESTINGGGGG #####
nu = filter_data[:,0] * 1e9
G = filter_data[:,1:]


#Filterbank
N_Filters = np.size(G,1)
N_nu = np.size(nu)
dnu = nu[1] - nu[0]


# # Generate yourself
# N_Filters = 70
# N_nu = 10001
# nu = np.linspace(200e9,400e9,N_nu,endpoint=True)
# dnu = nu(1) - nu(0)

## Background and atmosphere
p0 = 1e-18 # Background loading (use the full background loading later)

## Noise and integration time

## 1.275e-17 for R=100


NEPph = 1e-17
tau = 12.5 *2 * 4 # measurement time

sig_sq = NEPph**2 / (2 * tau)

Sigma = sig_sq * np.eye(N_Filters)


n_F0 = 10001
F0 = np.linspace(199e9,420e9,n_F0,endpoint=True)
R = 200


C = np.zeros((3,3,n_F0))

for i,F0_i in enumerate(F0):
    ## Line emission
    I0 = NEPph
    nu0 = F0_i
    delta_nu0 = nu0/R

    ## Galactic-like Lorentzian line shape
    # p = I0 * (0.5 * delta_nu0) / ( (nu - nu0)**2 + (0.5 * delta_nu0)**2 )
    # plt.plot(nu,p)
    # plt.yscale("log")
    # plt.grid(True,which="both")

    ## Partial derivatives of the line emission wrt the line emission parameters (make better into probability distribution)
    p_dI0 = (1/np.pi) * (0.5 * delta_nu0) / ( (nu - nu0)**2 + (0.5 * delta_nu0)**2 )
    p_dnu0 = (1/np.pi) * I0 * 2 * (0.5 * delta_nu0) * (nu - nu0) / (( (nu - nu0)**2 + (0.5 * delta_nu0)**2 )**2)
    p_ddelta_nu0 = (1/np.pi) * I0 * 0.5 *((nu - nu0)**2 - (0.5 * delta_nu0)**2)  / (( (nu - nu0)**2 + (0.5 * delta_nu0)**2 )**2)

    ## Sensitivities of the output signal with respect to the line emission parameters
    S_dI0 = G.T @ p_dI0 * dnu
    S_dnu0 = G.T @ p_dnu0 * dnu
    S_ddelta_nu0 = G.T @ p_ddelta_nu0 * dnu

    # M = np.stack((S_dI0,S_dnu0,S_ddelta_nu0),axis=1)
    M = np.stack((S_dI0,S_dnu0,S_ddelta_nu0),axis=1)

    ## Fisher information matrix
    F = M.T @ np.linalg.inv(Sigma) @ M

    ## Cramer-Rao bound
    C[:,:,i] = np.linalg.inv(F)

C_I0 = C[0,0,:]
C_nu0 = C[1,1,:]
C_delta_nu0 = C[2,2,:]








plt.figure(dpi=2000)
plt.plot(F0/1e9,np.sqrt(C_I0))
plt.ylim([0,3*I0])
plt.figure(dpi=2000)
plt.plot(F0/1e9,np.sqrt(C_nu0)/1e9)
plt.plot(F0/1e9,np.sqrt(C_delta_nu0)/1e9)
plt.ylim([0,15])

plt.figure(dpi=2000)
plt.plot(F0/1e9,I0/np.sqrt(C_I0))
# plt.ylim([0,3*I0])



class CramerRaoBound():
    """
    Cramer-Rao bound model

    """
