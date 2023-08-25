import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ioff()

from filterbank.components import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from filterbank.transformations import *
from filterbank.transformations import abcd_shuntload, chain,unchain,abcd2s
from filterbank.analysis import *







fig_path = "./figures/"

nF = int(1e5)
f = np.linspace(200e9,500e9,nF)

nF = int(5e2)
f2 = np.linspace(345e9,355e9,nF)

f0_min = 220e9
f0_max = 440e9


R_spacings = [300, 350, 400, 450, 500]
R_filter_widths = [300, 350, 400, 450, 500]

R_spacing_i = 0
R_filter_width_i = 0

Ql = R_spacings[R_spacing_i] #Ql 
S_os = R_filter_widths[R_filter_width_i] # oversampling

## Variances
sigma_Ql = 0#0.2
sigma_f0 = 0#0.1


Z0_res = 80
eps_eff_res = 40
Qi_res = 1120


Z0_thru = 80
eps_eff_thru = 40

TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

TL_res = TransmissionLine(Z0_res,eps_eff_res,Qi=Qi_res)

TransmissionLinesDict = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_thru
}

# For Ql is 500 target:
Ql_compensation = {
    'ManifoldFilter' : Ql*0.88,
    'ReflectorFilter' : Ql*1,
    'DirectionalFilter' : Ql*0.97
}

Filtertype_color_dict = {
    'ManifoldFilter' : np.array([86,199,74])/255,
    'ReflectorFilter' : np.array([242,131,45])/255,
    'DirectionalFilter' : np.array([90,136,237])/255
}

filterbank = dict()
Q_data = dict()
f0_data = dict()

Filters = (ManifoldFilter,DirectionalFilter,)

for Filter in Filters:
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=350e9,Ql=Ql,TransmissionLines=TransmissionLinesDict,sigma_f0=sigma_f0,sigma_Ql=sigma_Ql)

    isolated_filter.S(f2)

    isolated_filter.plot()

    # savestr = fig_path + filt_name + "_isolated_filter.svg"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)
    # plt.close()

    filterbank[filt_name] = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql,
        sigma_f0=sigma_f0,
        sigma_Ql=sigma_Ql
    )

    filterbank[filt_name].S(f)

    # filterbank[filt_name].plot()

    # savestr = fig_path + filt_name + "_filterbank.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)

    f0_data[filt_name],Q_data[filt_name],_,_ = filterbank[filt_name].realized_parameters()

    plt.close()


filterbank = dict()
envelope = dict()
Q_compensated_data = dict()
f0_compensated_data = dict()
inband_filter_eff = dict()
average_env_eff = dict()

######## Compensated ################
for Filter in Filters:
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=350e9,Ql=Ql_compensation[filt_name],TransmissionLines=TransmissionLinesDict,sigma_f0=sigma_f0,sigma_Ql=sigma_Ql)

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
        Ql=Ql_compensation[filt_name],
        sigma_f0=sigma_f0,
        sigma_Ql=sigma_Ql
    )

    filterbank[filt_name].S(f)

    # filterbank[filt_name].plot()

    # savestr = fig_path + filt_name + "_filterbank.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)

    f0_compensated_data[filt_name],Q_compensated_data[filt_name],inband_filter_eff[filt_name],_ = filterbank[filt_name].realized_parameters()
    average_env_eff[filt_name],_ = filterbank_analysis(Filterbank=filterbank[filt_name],f0_realized=f0_compensated_data[filt_name],Ql_realized=Q_compensated_data[filt_name])
    

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

savestr = fig_path + "Realized_Q-factors.svg"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


###############################################
fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(2.5,2.5),layout='constrained')

for Filter in Filters:
    filt_name = str(Filter.__name__)
    ax.scatter(f0_compensated_data[filt_name]/1e9,inband_filter_eff[filt_name],label=f"{filt_name}",color=Filtertype_color_dict[filt_name])
    ax.plot(f0_compensated_data[filt_name]/1e9, np.mean(inband_filter_eff[filt_name])*np.ones(len(f0_compensated_data[filt_name])),color=Filtertype_color_dict[filt_name])
    ax.plot(f0_compensated_data[filt_name]/1e9, average_env_eff[filt_name]*np.ones(len(f0_compensated_data[filt_name])),color=Filtertype_color_dict[filt_name],linestyle=":")    
    



ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('efficiency')  # Add a y-label to the axes.
ax.set_title("In-band filter efficiency")  # Add a title to the axes.
# plt.ylim(-30,0)
# plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = fig_path + "In-band_filter_efficiency.svg"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




################################################
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
# ax.legend();  # Add a legend.

savestr = fig_path + "Filterbank_DirectionalFilter.svg"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




#####################################
fig, ax =plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(6,1.5),layout='constrained')

S31_all = filterbank["ManifoldFilter"].S31_absSq_list

#ax
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_all)[0])

for i,S31_absSq in enumerate(S31_all):
    ax.plot(f/1e9,100*S31_absSq,color=cmap(norm(i)))

sum_filters = np.sum(S31_all,axis=0)
ax.plot(f/1e9,100*sum_filters,label='sum filters',color="0.5")

envelope["ManifoldFilter"] = np.array(filterbank['ManifoldFilter'].S31_absSq_list).max(axis=0)
# ax.plot(f/1e9,100*envelope["DirectionalFilter"],label='envelope',color=(0.,0.,0.))

ax.plot(f/1e9,100*filterbank["ManifoldFilter"].S11_absSq,label='S11',color=(0.,1.,1.))
ax.plot(f/1e9,100*filterbank["ManifoldFilter"].S21_absSq,label='S21',color=(1.,0.,1.))

ax.set_ylim(0,100)
ax.set_xlim(np.min(f)/1e9,np.max(f)/1e9)
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Transmission [%]')  # Add a y-label to the axes.
# plt.title("Realized Q-factors")  # Add a title to the axes.
# ax.legend();  # Add a legend.

savestr = fig_path + "Filterbank_ManifoldFilter.svg"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)

plt.show()