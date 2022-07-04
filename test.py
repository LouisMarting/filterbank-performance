import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.ioff()

from filterbank import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from transformations import *
from transformations import abcd_shuntload, chain,unchain,abcd2s



nF = int(5e3)
f = np.linspace(80,440,nF)

f0_min = 100
f0_max = 400
Ql = 25

Z0_res = 22.2
eps_eff_res = 70.7

Z0_thru = 72.8
eps_eff_thru = 42.9

TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

TL_res = TransmissionLine(Z0_res,eps_eff_res)

TransmissionLines = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_thru
}

TestDirFilter =DirectionalFilter(200e9,Ql,TransmissionLines=TransmissionLines)

ABCD_dir = TestDirFilter.ABCD(f)

ABCD_chain = chain(ABCD_dir,ABCD_dir)

ABCD_unchained = unchain(ABCD_chain,ABCD_dir)

print(np.all(np.isclose(ABCD_unchained,ABCD_dir)))





#=============================================
var_settings = [(0,0)]#[(0,0), (0.1,0.05), (0.2,0.1), (0.3,0.3)]
for var_setting in var_settings:
    OmegaFilterbank = Filterbank(DirectionalFilter,TransmissionLines,f0_min=f0_min,f0_max=f0_max,Ql=Ql, sigma_f0=var_setting[0],sigma_Ql=var_setting[1])

    S = OmegaFilterbank.S(f)


    S11_absSq = np.abs(S[0][0][0])**2
    S21_absSq = np.abs(S[0][1][0])**2

    S31_absSq_list = []
    for i in np.arange(1,np.shape(S)[0],2):
        S_filt1 = np.abs(S[i][1][0])**2
        S_filt2 = np.abs(S[i+1][1][0])**2
        
        S31_absSq_list.append(S_filt1 + S_filt2)

    fig, ax =plt.subplots(figsize=(12,5),layout='constrained')

    cmap = mpl.cm.get_cmap('rainbow').copy()
    norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_absSq_list)[0])

    for i,S31_absSq in enumerate(S31_absSq_list):
        ax.plot(f,10*np.log10(S31_absSq),color=cmap(norm(i)))

    ax.plot(f,10*np.log10(S11_absSq),label='S11',color=(0.,1.,1.))
    ax.plot(f,10*np.log10(S21_absSq),label='S21',color=(1.,0.,1.))

    ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
    ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
    ax.set_title("Filterbank")  # Add a title to the axes.
    ax.legend();  # Add a legend.
    plt.ylim(-30,0)
    plt.show()



# #===================================================================
# GammaFilterbank = Filterbank(ReflectorFilter,TransmissionLines,f0_min=f0_min,f0_max=f0_max,Ql=Ql)

# S = GammaFilterbank.S(f)


# S11_absSq = np.abs(S[0][0][0])**2
# S21_absSq = np.abs(S[0][1][0])**2

# S31_absSq_list = []
# for i in np.arange(1,np.shape(S)[0],1):
#     S31_absSq = np.abs(S[i][1][0])**2

#     S31_absSq_list.append(S31_absSq)

# fig, ax =plt.subplots(figsize=(12,5),layout='constrained')
# ax.plot(f,10*np.log10(S11_absSq),label='S11')
# ax.plot(f,10*np.log10(S21_absSq),label='S21')

# cmap = mpl.cm.get_cmap('rainbow').copy()
# norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_absSq_list)[0])

# for i,S31_absSq in enumerate(S31_absSq_list):
#     ax.plot(f,10*np.log10(S31_absSq),color=cmap(norm(i)))

# ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
# ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
# ax.set_title("Filterbank")  # Add a title to the axes.
# ax.legend();  # Add a legend.
# plt.ylim(-30,0)
# plt.show()

# #===================================================================
# IotaFilterbank = Filterbank(ManifoldFilter,TransmissionLines,f0_min=f0_min,f0_max=f0_max,Ql=Ql)

# S = IotaFilterbank.S(f)


# S11_absSq = np.abs(S[0][0][0])**2
# S21_absSq = np.abs(S[0][1][0])**2

# S31_absSq_list = []
# for i in np.arange(1,np.shape(S)[0],1):
#     S31_absSq = np.abs(S[i][1][0])**2

#     S31_absSq_list.append(S31_absSq)

# fig, ax =plt.subplots(figsize=(12,5),layout='constrained')
# ax.plot(f,10*np.log10(S11_absSq),label='S11')
# ax.plot(f,10*np.log10(S21_absSq),label='S21')

# cmap = mpl.cm.get_cmap('rainbow').copy()
# norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_absSq_list)[0])

# for i,S31_absSq in enumerate(S31_absSq_list):
#     ax.plot(f,10*np.log10(S31_absSq),color=cmap(norm(i)))

# ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
# ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
# ax.set_title("Filterbank")  # Add a title to the axes.
# ax.legend();  # Add a legend.
# plt.ylim(-30,0)
# plt.show()


###=================================================================================
# Filter : ReflectorFilter = GammaFilterbank.Filters[0]

# ABCD = Filter.Reflector.ABCD(f)

# S_filter = abcd2s(ABCD,Z0_thru)

# S11_absSq = np.abs(S_filter[0][0])**2
# S21_absSq = np.abs(S_filter[1][0])**2


# ABCD_termination = np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)
# ABCD = Filter.Reflector.ABCD(f)

# S_filter_31 = abcd2s(ABCD,Z0_thru)

# S31_absSq = np.abs(S_filter_31[1][0])**2


# fig, ax =plt.subplots(figsize=(12,5),layout='constrained')

# ax.plot(f,10*np.log10(S11_absSq),label='S11')
# ax.plot(f,10*np.log10(S21_absSq),label='S21')
# # ax.plot(f,10*np.log10(S31_absSq,),label='S31')


# ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
# ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
# ax.set_title("Filterbank")  # Add a title to the axes.
# ax.legend();  # Add a legend.
# plt.ylim(-30,0)
# plt.show()

# ABCD_succeeding = OmegaFilterbank.Filters[0].ABCD_sep(f)

# for i, Filter in enumerate(OmegaFilterbank.Filters):
#     Filter : BaseFilter # set the expected datatype of Filter

#     ABCD_succeeding = chain(
#         ABCD_succeeding,
#         Filter.ABCD(f),
#         Filter.ABCD_sep(f) 
#     )


# S_filter = abcd2s(ABCD_succeeding,Z0_thru)

# S11_absSq = np.abs(S_filter[0][0])**2
# S21_absSq = np.abs(S_filter[1][0])**2


# fig, ax =plt.subplots(figsize=(12,5),layout='constrained')

# ax.plot(f,10*np.log10(S11_absSq),label='S11')
# ax.plot(f,10*np.log10(S21_absSq),label='S21')

# ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
# ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
# ax.set_title("Filterbank")  # Add a title to the axes.
# ax.legend();  # Add a legend.
# plt.ylim(-30,0)
# plt.show()