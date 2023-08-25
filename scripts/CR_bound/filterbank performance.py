import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from deshima_sensitivity.instruments import photon_NEP_kid
from deshima_sensitivity.simulator import spectrometer_sensitivity
from deshima_sensitivity.physics import h, e
### import csv files 
# filter_data = np.genfromtxt('Directional_filter_data.csv',delimiter=',')
# filter_data = np.genfromtxt('Manifold_filter_data.csv',delimiter=',')

# data_files = ['Manifold_filter_data.csv','Directional_filter_data.csv']
data_files = ['Directional_filter_data.csv']
R_line_list = [50,100,200,800]

for data_file in data_files:
    filter_data = np.genfromtxt(data_file,delimiter=',')


    # constants
    Delta_Al = 188.0 * 10 ** -6 * e  # gap energy of Al
    eta_pb = 0.4  # Pair breaking efficiency



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





    F0 = nu
    n_F0 = N_nu
    for R_line in R_line_list:
        # R_line = 50

        # Set R = 0 to get exact atmosphere transmission
        Dir_filter_chip = spectrometer_sensitivity(F=F0,R=0,pwv=0.5,EL=60,eta_circuit=0.8)
        psd_KID = Dir_filter_chip["psd_KID"]
        Pkid = psd_KID @ G * dnu


        # NEP from psd
        poisson_term = ( (2 * h * nu) * psd_KID ) @ G * dnu
        bunching_term = 2 * (psd_KID @ G)**2 * dnu
        gr_term = 4 * Delta_Al * (psd_KID @ G * dnu) / eta_pb

        NEP = np.sqrt(poisson_term + bunching_term + gr_term)

        # plt.figure(num=5,dpi=2000)
        # plt.plot(np.flip(NEP**2))
        # plt.plot(np.flip(poisson_term))
        # plt.plot(np.flip(bunching_term))
        # plt.plot(np.flip(gr_term))
        # plt.yscale("log")

        # NEP = np.ones(np.size(NEP)) * 1e-16

        tau = 8 * 60 * 60 # measurement time

        sig_sq = NEP**2 / (2 * tau)

        Sigma = np.diag(sig_sq)


        C = np.zeros((3,3,n_F0))

        for i,F0_i in enumerate(F0):
            ## Line emission
            I0 = 1e-17
            nu0 = F0_i
            delta_nu0 = nu0/R_line

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








        plt.figure(num=1,dpi=2000)
        plt.plot(F0/1e9,np.sqrt(C_I0))
        plt.ylim([0,3*I0])
        plt.figure(num=2,dpi=2000)
        plt.plot(F0/1e9,np.sqrt(C_nu0)/1e9)
        plt.ylim([0,15])
        plt.figure(num=3,dpi=2000)
        plt.plot(F0/1e9,np.sqrt(C_delta_nu0)/1e9)
        plt.ylim([0,15])

        plt.figure(num=4,dpi=2000)
        plt.plot(F0/1e9,I0/np.sqrt(C_I0))
        # plt.ylim([0,3*I0])

    


class CramerRaoBound():
    """
    Cramer-Rao bound model

    """
