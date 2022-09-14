import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ioff()

from filterbank import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from transformations import *
from transformations import abcd_shuntload, chain,unchain,abcd2s
from analysis import *







fig_path = "H:/My Documents/Thesis/SZ-mapping filterbank concept for Kenichi/"
# fig_path = "W:/student-homes/m/lmarting/My Documents/Thesis/Figures/Thesis/Report figures/"

nF = int(5e3)
f = np.linspace(100e9,330e9,nF)

nF = int(5e2)
f2 = np.linspace(190e9,210e9,nF)

f0_min = 135e9
f0_max = 270e9
Ql = 10

Z0_res = 80
eps_eff_res = 40

Z0_thru = 80
eps_eff_thru = 40

TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

TL_res = TransmissionLine(Z0_res,eps_eff_res)

TransmissionLinesDict = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_thru
}


Ql_compensation = {
    'ManifoldFilter' : Ql*1.30,
    'ReflectorFilter' : Ql*0.7,
    'DirectionalFilter' : Ql*1.12
}

Filtertype_color_dict = {
    'ManifoldFilter' : np.array([86,199,74])/255,
    'ReflectorFilter' : np.array([242,131,45])/255,
    'DirectionalFilter' : np.array([90,136,237])/255
}

filterbank = dict()
Q_data = dict()
f0_data = dict()

Filters = (DirectionalFilter,)

for Filter in Filters:
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=200e9,Ql=Ql,TransmissionLines=TransmissionLinesDict)

    isolated_filter.S(f2)

    isolated_filter.plot()

    savestr = fig_path + filt_name + "_isolated_filter.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(savestr)
    # plt.close()

    filterbank[filt_name] = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql
    )

    filterbank[filt_name].S(f)

    # filterbank[filt_name].plot()

    # savestr = fig_path + filt_name + "_filterbank.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)

    f0_data[filt_name],Q_data[filt_name] = filterbank[filt_name].realized_parameters()

    plt.close()


filterbank = dict()
envelope = dict()
Q_compensated_data = dict()
f0_compensated_data = dict()

######## Compensated ################
for Filter in Filters:
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=200e9,Ql=Ql_compensation[filt_name],TransmissionLines=TransmissionLinesDict)

    isolated_filter.S(f2)

    # isolated_filter.plot()

    # savestr = fig_path + filt_name + "_isolated_filter.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)
    # plt.close()

    filterbank[filt_name] = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql_compensation[filt_name]
    )

    filterbank[filt_name].S(f)

    # filterbank[filt_name].plot()

    # savestr = fig_path + filt_name + "_filterbank.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)

    f0_compensated_data[filt_name],Q_compensated_data[filt_name] = filterbank[filt_name].realized_parameters()
    

    plt.close()

fig, (ax1, ax2) =plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(5.5,2.5),layout='constrained')

for Filter in Filters:
    filt_name = str(Filter.__name__)
    ax1.scatter(f0_data[filt_name]/1e9,Q_data[filt_name],label=f"{filt_name}",color=Filtertype_color_dict[filt_name])
    ax2.scatter(f0_compensated_data[filt_name]/1e9,Q_compensated_data[filt_name],label=f"{filt_name}",color=Filtertype_color_dict[filt_name])


ax1.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax2.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax1.set_ylabel('Q-factor')  # Add a y-label to the axes.
ax1.set_title("Uncompensated Q-factors")  # Add a title to the axes.
ax1.legend();  # Add a legend.
ax2.set_title("Compensated Q-factors")  # Add a title to the axes.
# plt.ylim(-30,0)
# plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = fig_path + "Realized_Q-factors.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




fig, ax =plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(6,1.5),layout='constrained')

S31_all = filterbank["DirectionalFilter"].S31_absSq_list

#ax
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_all)[0])

for i,S31_absSq in enumerate(S31_all):
    ax.plot(f/1e9,100*S31_absSq,color=cmap(norm(i)))

sum_filters = np.sum(S31_all,axis=0)
ax.plot(f/1e9,100*sum_filters,label='sum filters',color="0.5")

envelope["DirectionalFilter"] = np.array(filterbank['DirectionalFilter'].S31_absSq_list).max(axis=0)
# ax.plot(f/1e9,100*envelope["DirectionalFilter"],label='envelope',color=(0.,0.,0.))

ax.plot(f/1e9,100*filterbank["DirectionalFilter"].S11_absSq,label='S11',color=(0.,1.,1.))
ax.plot(f/1e9,100*filterbank["DirectionalFilter"].S21_absSq,label='S21',color=(1.,0.,1.))

ax.set_ylim(0,100)
ax.set_xlim(np.min(f)/1e9,np.max(f)/1e9)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Transmission [%]')  # Add a y-label to the axes.
# plt.title("Realized Q-factors")  # Add a title to the axes.
ax.legend();  # Add a legend.

savestr = fig_path + "Filterbank_SZ_mapping.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)

plt.show()