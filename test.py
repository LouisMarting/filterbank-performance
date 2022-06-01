import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.ioff()

from filterbank import Resonator,TransmissionLine,DirectionalFilter,Filterbank,BaseFilter
from transformations import *
from transformations import abcd_shuntload, chain,unchain,abcd2s



nF = int(1e2)
f = np.linspace(199e9,201e9,nF)


f0 = 200e9
Ql = 100

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

Res = Resonator(f0,Ql,TL_res,Z0_thru/2)

ABCD_res = Res.ABCD(f)

print(ABCD_res[:,:,0])

TestDirFilter =DirectionalFilter(f0,Ql,TransmissionLines=TransmissionLines)

ABCD_dir = TestDirFilter.ABCD(f)

ABCD_chain = chain(ABCD_dir,ABCD_dir)

ABCD_unchained = unchain(ABCD_chain,ABCD_dir)

print(np.all(np.isclose(ABCD_unchained,ABCD_dir)))

f0_min = 200
f0_max = 400
OmegaFilterbank = Filterbank(DirectionalFilter,TransmissionLines,f0_min=f0_min,f0_max=f0_max,Ql=Ql)

f = np.linspace(180,440,int(1e3))
S = OmegaFilterbank.S(f)



S11_absSq = np.abs(S[0][0][0])**2
S21_absSq = np.abs(S[0][1][0])**2

S31_absSq_list = []
for i in np.arange(1,np.shape(S)[0],2):
    S_filt1 = np.abs(S[i][1][0])**2
    S_filt2 = np.abs(S[i+1][1][0])**2
    
    S31_absSq_list.append(S_filt1 + S_filt2)

fig, ax =plt.subplots(figsize=(12,5),layout='constrained')
ax.plot(f,10*np.log10(S11_absSq),label='S11')
ax.plot(f,10*np.log10(S21_absSq),label='S21')

cmap = mpl.cm.get_cmap('rainbow').copy().reversed()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_absSq_list)[0])

for i,S31_absSq in enumerate(S31_absSq_list):
    ax.plot(f,10*np.log10(S31_absSq),color=cmap(norm(i)))

ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
ax.set_title("Filterbank")  # Add a title to the axes.
ax.legend();  # Add a legend.
plt.ylim(-30,0)
plt.show()



Filter : DirectionalFilter = OmegaFilterbank.Filters[26]

ABCD = Filter.ABCD(f)


S_filter = abcd2s(ABCD,Z0_thru)

S11_absSq = np.abs(S_filter[0][0])**2
S21_absSq = np.abs(S_filter[1][0])**2


fig, ax =plt.subplots(figsize=(12,5),layout='constrained')

ax.plot(f,10*np.log10(S11_absSq),label='S11')
ax.plot(f,10*np.log10(S21_absSq),label='S21')

ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
ax.set_title("Filterbank")  # Add a title to the axes.
ax.legend();  # Add a legend.
plt.ylim(-30,0)
plt.show()

ABCD_succeeding = np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)

for i, Filter in enumerate(OmegaFilterbank.Filters):
    Filter : BaseFilter # set the expected datatype of Filter
    

    # Eventually, these indexed lists could be replaced by cached versions.

    ABCD_succeeding = chain(
        ABCD_succeeding,
        Filter.ABCD(f),
        Filter.ABCD_sep(f) # Can we use np.insert() for these and do this calc faster outside of this for loop?
    )

ABCD_succeeding = chain(OmegaFilterbank.Filters[0].ABCD(f),ABCD_succeeding)


S_filter = abcd2s(ABCD_succeeding,[Z0_thru, Z0_thru])

S11_absSq = np.abs(S_filter[0][0])**2
S21_absSq = np.abs(S_filter[1][0])**2


fig, ax =plt.subplots(figsize=(12,5),layout='constrained')

ax.plot(f,10*np.log10(S11_absSq),label='S11')
ax.plot(f,10*np.log10(S21_absSq),label='S21')

ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('S-params [dB]')  # Add a y-label to the axes.
ax.set_title("Filterbank")  # Add a title to the axes.
ax.legend();  # Add a legend.
plt.ylim(-30,0)
plt.show()