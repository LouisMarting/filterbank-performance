import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
plt.ioff()

from filterbank.components import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from filterbank.transformations import *
from filterbank.transformations import abcd_shuntload, chain,unchain,abcd2s
from filterbank.analysis import *

thesisfig_path = "H:/My Documents/Thesis/Figures/Thesis/Report figures/"
# thesisfig_path = "W:/student-homes/m/lmarting/My Documents/Thesis/Figures/Thesis/Report figures/"

nF = int(5e3)
f = np.linspace(180e9,440e9,nF)

nF = int(5e2)
f2 = np.linspace(290e9,310e9,nF)

f0_min = 200e9
f0_max = 400e9
Ql = 100

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

Filters = (ManifoldFilter,ReflectorFilter,DirectionalFilter)

Filtertype_color_dict = {
    'ManifoldFilter' : np.array([86,199,74])/255,
    'ReflectorFilter' : np.array([242,131,45])/255,
    'DirectionalFilter' : np.array([90,136,237])/255
}

legend_filter_name = {
    'ManifoldFilter' : 'Manifold Filter',
    'ReflectorFilter' : 'Reflector Filter',
    'DirectionalFilter' : 'Directional Filter'
}




filterbank = dict()
Q_data = dict()
f0_data = dict()

########## Uncompensated ##############
for Filter in Filters:
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=300e9,Ql=Ql,TransmissionLines=TransmissionLinesDict,compensate=False)

    isolated_filter.S(f2)

    isolated_filter.plot()

    savestr = thesisfig_path + filt_name + "_isolated_filter.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(savestr)
    # plt.close()

    filterbank[filt_name] = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql,
        compensate=False
    )

    filterbank[filt_name].S(f)

    # filterbank[filt_name].plot()

    # savestr = thesisfig_path + filt_name + "_filterbank.pdf"
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
inband_fraction = dict()

######## Compensated ################
for Filter in (ManifoldFilter,ReflectorFilter,DirectionalFilter):
    filt_name = str(Filter.__name__)

    isolated_filter : BaseFilter = Filter(f0=300e9,Ql=Ql,TransmissionLines=TransmissionLinesDict)

    isolated_filter.S(f2)

    # isolated_filter.plot()

    # savestr = thesisfig_path + filt_name + "_isolated_filter.pdf"
    # plt.savefig(fname=savestr)
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title(savestr)
    # plt.close()

    filterbank[filt_name] = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql
    )

    filterbank[filt_name].S(f)

    filterbank[filt_name].plot()

    savestr = thesisfig_path + filt_name + "_filterbank.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(savestr)

    f0_compensated_data[filt_name],Q_compensated_data[filt_name],inband_filter_eff[filt_name],inband_fraction[filt_name] = filterbank[filt_name].realized_parameters()
    

    plt.close()

fig, (ax1, ax2) =plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(5.5,2.5),layout='constrained')

for Filter in Filters:
    filt_name = str(Filter.__name__)
    ax1.scatter(f0_data[filt_name]/1e9,Q_data[filt_name],label=legend_filter_name[filt_name],color=Filtertype_color_dict[filt_name])
    ax2.scatter(f0_compensated_data[filt_name]/1e9,Q_compensated_data[filt_name],label=legend_filter_name[filt_name],color=Filtertype_color_dict[filt_name])


ax1.set_xlim(195,405)
ax2.set_xlim(195,405)
ax1.set_xlabel('Frequency [GHz]')  # Add an x-label to the axes.
ax2.set_xlabel('Frequency [GHz]')  # Add an x-label to the axes.
ax1.set_ylabel('Q-factor')  # Add a y-label to the axes.
ax1.set_title("Uncompensated Q-factors")  # Add a title to the axes.
ax1.legend();  # Add a legend.
ax2.set_title("Compensated Q-factors")  # Add a title to the axes.
# plt.ylim(-30,0)
# plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = thesisfig_path + "Realized_Q-factors.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)





fig, (ax1, ax2, ax3, ax4) =plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(5.5,5),layout='constrained')

S31_all = filterbank["DirectionalFilter"].S31_absSq_list

#ax1
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_all)[0])

for i,S31_absSq in enumerate(S31_all):
    ax1.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

sum_filters = np.sum(S31_all,axis=0)
ax1.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")

# ax1.set_yscale("log")
ax1.set_ylim(0,100)
ax1.set_title("Example of Filter Responses")
ax1.legend(loc='upper right')

#ax2-4
envelope["ManifoldFilter"] = np.array(filterbank['ManifoldFilter'].S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank['ManifoldFilter'].S31_absSq_list,axis=0)
ax2.plot(f/1e9,envelope["ManifoldFilter"]*100,label=r'$\eta_{env}$',color=(0.,0.,0.))
ax2.plot(f/1e9,sum_filters*100,color="0.5")
ax2.plot(f/1e9,filterbank["ManifoldFilter"].S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax2.plot(f/1e9,filterbank["ManifoldFilter"].S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))

# ax2.set_yscale("log")
ax2.set_ylim(0,100)
ax2.set_title("Manifold Filter - Filterbank")
ax2.legend(loc='upper right')
#---

envelope["ReflectorFilter"] = np.array(filterbank['ReflectorFilter'].S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank['ReflectorFilter'].S31_absSq_list,axis=0)
ax3.plot(f/1e9,envelope["ReflectorFilter"]*100,label='envelope',color=(0.,0.,0.))
ax3.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax3.plot(f/1e9,filterbank["ReflectorFilter"].S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax3.plot(f/1e9,filterbank["ReflectorFilter"].S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))

# ax3.set_yscale("log")
ax3.set_ylim(0,100)
ax3.set_title("Reflector Filter - Filterbank")
#---

envelope["DirectionalFilter"] = np.array(filterbank['DirectionalFilter'].S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank['DirectionalFilter'].S31_absSq_list,axis=0)
ax4.plot(f/1e9,envelope["DirectionalFilter"]*100,label='envelope',color=(0.,0.,0.))
ax4.plot(f/1e9,sum_filters*100,label='sum filters',color="0.5")
ax4.plot(f/1e9,filterbank["DirectionalFilter"].S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax4.plot(f/1e9,filterbank["DirectionalFilter"].S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))

# ax4.set_yscale("log")
ax4.set_ylim(0,100)
ax4.set_title("Directional Filter - Filterbank")
ax4.set_xlabel('Frequency [GHz]')
#---

# fig.supxlabel('Frequency [GHz]')  # Add an x-label to the axes.
fig.supylabel('Transmission [%]')  # Add a y-label to the axes.
# plt.title("Realized Q-factors")  # Add a title to the axes.
# ax1.legend();  # Add a legend.
# plt.yscale("log")
# plt.ylim(1,100)
plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = thesisfig_path + "Filterbanks_envelope.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)



## inband filter eff
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')
for Filter in Filters:
    filt_name = str(Filter.__name__)
    # ax.plot(f/1e9,envelope[filt_name]*100,color=Filtertype_color_dict[filt_name],alpha=0.3)
    ax.scatter(f0_compensated_data[filt_name]/1e9,inband_filter_eff[filt_name]*100,label=legend_filter_name[filt_name],color=Filtertype_color_dict[filt_name])
    

ax.set_ylim(0,100)
ax.set_xlim(195,405)
ax.set_xlabel('Frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('Efficiency [%]')  # Add a y-label to the axes.
ax.set_title("In-Band Filter Efficiency")  # Add a title to the axes.
ax.legend();  # Add a legend.

savestr = thesisfig_path + "Inband_filter_eff.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)

## inband fraction
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')
for Filter in Filters:
    filt_name = str(Filter.__name__)
    ax.scatter(f0_compensated_data[filt_name]/1e9,inband_fraction[filt_name]*100,label=legend_filter_name[filt_name],color=Filtertype_color_dict[filt_name])

ax.set_ylim(0,100)
ax.set_xlim(195,405)
ax.set_xlabel('Frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('Fraction [%]')  # Add a y-label to the axes.
ax.set_title("In-Band Fraction")  # Add a title to the axes.
ax.legend();  # Add a legend.


savestr = thesisfig_path + "Inband_fraction.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




#=============================================
var_settings = [(0.0,0.0),(0.12,0.07), (0.3,0.15)] # [(0.12,0.07), (0.3,0.14), (0.7,0.28), (0.2,0), (0,0.2)]
n_runs = 20
avg_envelope_eff = np.empty((np.size(var_settings,axis=0),len(Filters),n_runs))
usable_spectral_fraction = np.empty((np.size(var_settings,axis=0),len(Filters),n_runs))

var_settings_colors = [
    np.array([90,136,237])/255,
    np.array([86,199,74])/255,
    np.array([242,131,45])/255,
    np.array([132,82,201])/255,
    np.array([108,98,96])/255
]



fig, axes =plt.subplots(nrows=len(Filters),ncols=2,sharex=True,sharey=True,figsize=(4,5),layout='constrained')
for v_i,var_setting in enumerate(var_settings):
    for i,row in enumerate(axes):
        filt_name = str(Filters[i].__name__)
        filterbank = Filterbank(Filters[i],TransmissionLinesDict,f0_min=f0_min,f0_max=f0_max,Ql=Ql, sigma_f0=var_setting[0],sigma_Ql=var_setting[1])

        # OmegaFilterbank.plot()
        f0_realized, Ql_realized, df_variance, Ql_variance, avg_envelope_eff[v_i][i], usable_spectral_fraction[v_i][i] = analyse_variance(filterbank,f,n_filterbanks=n_runs)

        hist_data = [df_variance,Ql_variance] #to percentages

        for j,ax in enumerate(row):
            ax.hist(hist_data[j]*100,bins=np.linspace(-100,100,num=41),facecolor=var_settings_colors[v_i],alpha=0.4)



custom_lines = [Line2D([0], [0], color=var_settings_colors[0], lw=4),
                Line2D([0], [0], color=var_settings_colors[1], lw=4),
                Line2D([0], [0], color=var_settings_colors[2], lw=4)]

axes[0,1].legend(custom_lines, [r"$\sigma_{f_i} = 0 \, \%, \: \sigma_{Q_\mathrm{L}} = 0\, \%$",r"$\sigma_{f_i} = 12 \, \%, \: \sigma_{Q_\mathrm{L}} = 7\, \%$",r"$\sigma_{f_i} = 30 \, \%, \: \sigma_{Q_\mathrm{L}} = 15\, \%$"], loc='upper right')
# axes[0,1].legend(custom_lines, [r"$Q_\mathrm{i} = \infty$",r"$Q_\mathrm{i} = 500$",r"$Q_\mathrm{i} = 200$"], loc='upper right')

axes[0,0].set_title("Frequency Scatter")
axes[0,1].set_title("Ql Scatter")
axes[-1,0].set_xlabel(r"$e_{f_i}$ [%]")
axes[-1,1].set_xlabel(r"$e_{Q_\mathrm{L}}$ [%]")
axes[0,0].set_ylabel("Manifold filter")
axes[1,0].set_ylabel("Reflector filter")
axes[2,0].set_ylabel("Directional filter")
axes[0,0].set_xlim(-100,100)
axes[0,1].set_xlim(-100,100)
axes[0,0].set_ylim(0,200*(n_runs/5))
axes[0,1].set_ylim(0,200*(n_runs/5))

savestr = thesisfig_path + f"Histogram_frequency_and_Ql_scatter.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)






## BOXPLOT 

ticks = ["Manifold\nFilter","Reflector\nFilter","Directional\nFilter"]

fig, (ax1,ax2) =plt.subplots(nrows=1,ncols=2,figsize=(5.5,2.5),layout='constrained')

positions_base = np.array(np.arange(len(ticks)))

positions_offset = np.linspace(-0.25,0.25,len(var_settings))

for v_i,var_setting in enumerate(var_settings):

    avg_envelope_eff_plot = ax1.boxplot(
        avg_envelope_eff[v_i].T*100,
        positions = positions_base + positions_offset[v_i],
        widths= 0.5 / len(var_settings)
        )
    usable_spectral_fraction_plot = ax2.boxplot(
        usable_spectral_fraction[v_i].T*100,
        positions = positions_base + positions_offset[v_i],
        widths= 0.5 / len(var_settings)
        )
    
    def define_box_properties(plot_name, color_code):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
            plt.setp(plot_name['fliers'], markeredgecolor=color_code,markersize=4)
    
    define_box_properties(avg_envelope_eff_plot,var_settings_colors[v_i])
    define_box_properties(usable_spectral_fraction_plot,var_settings_colors[v_i])


ax1.set_xticks(positions_base,ticks)
# ax1.set_ylim(0,100)
ax1.yaxis.grid(True)
ax1.set_title("Average Envelope Efficiency")
ax1.set_ylabel('Efficiency [%]')
ax1.tick_params(axis='x',which='minor',bottom=False,top=False)

ax2.set_xticks(positions_base,ticks)
# ax2.set_ylim(0,100)
ax2.yaxis.grid(True)
ax2.set_title("Usable Spectral Fraction")
ax2.set_ylabel('Fraction [%]')
ax2.tick_params(axis='x',which='minor',bottom=False,top=False)


custom_lines = [Line2D([0], [0], color=var_settings_colors[0], lw=4),
                Line2D([0], [0], color=var_settings_colors[1], lw=4),
                Line2D([0], [0], color=var_settings_colors[2], lw=4)]

ax1.legend(custom_lines, [r"$\sigma_{f_i} = 0 \, \%, \: \sigma_{Q_\mathrm{L}} = 0\, \%$",r"$\sigma_{f_i} = 12 \, \%, \: \sigma_{Q_\mathrm{L}} = 7\, \%$",r"$\sigma_{f_i} = 30 \, \%, \: \sigma_{Q_\mathrm{L}} = 15\, \%$"], loc='lower right')
# ax1.legend(custom_lines, [r"$Q_\mathrm{i} = \infty$",r"$Q_\mathrm{i} = 500$",r"$Q_\mathrm{i} = 200$"], loc='lower right')

savestr = thesisfig_path + "Boxplot_average_env_eff_and_USF.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)








## Filterbank variance plot
filterbank_variance = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    sigma_f0=0.2,
    sigma_Ql=0.1
)

filterbank_variance.S(f)

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(5.5,1.5),layout='constrained')

S31_all = filterbank_variance.S31_absSq_list

sum_filters = np.sum(S31_all,axis=0)

#ax1
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_all)[0])

for i,S31_absSq in enumerate(S31_all):
    ax.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

ax.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")

# ax.set_yscale("log")
ax.set_ylim(0,100)
ax.set_title("Filterbank with Variances")
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Transmission [%]')
ax.legend(loc='upper right')

ax.set_xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = thesisfig_path + "Filterbank_variance.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)



for Filter in (ManifoldFilter,ReflectorFilter,DirectionalFilter):
    filt_name = str(Filter.__name__)

    filterbank = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql,
        sigma_f0=0.2,
        sigma_Ql=0.1
    )

    filterbank.S(f)

    filterbank.plot()

    savestr = thesisfig_path + filt_name + "_filterbank_variance.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(savestr)
    plt.close()



## Filterbank USF threshold plot
_, _, _, _, avg_envelope_eff, usable_spectral_fraction = analyse_variance(filterbank_variance,f,n_filterbanks=1)

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')

#ax1
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=20, vmax=35)

for i,S31_absSq in enumerate(S31_all):
    ax.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))



envelope = np.array(S31_all).max(axis=0)
sum_filters = np.sum(S31_all,axis=0)
ax.plot(f/1e9,envelope*100,label=r'$\eta_\mathrm{env}$',color=(0.,0.,0.),linestyle=":",linewidth=0.75)
# ax.plot(f/1e9,sum_filters*100,label='sum filters',color="0.5")
ax.hlines(avg_envelope_eff*100,np.min(f)/1e9,np.max(f)/1e9,label=r'$\overline{\eta_\mathrm{env}}$',colors=[(0.,0.,0.,0.5)],linewidth=0.6)
ax.hlines(avg_envelope_eff*100/np.sqrt(2),np.min(f)/1e9,np.max(f)/1e9,label=r'$\mathrm{USF}$ threshold',colors=[(1.,0.,0.,0.5)],linewidth=0.6,linestyles=["dashed"])
# ax.hlines(avg_envelope_eff*100*np.sqrt(2),np.min(f)/1e9,np.max(f)/1e9,colors=(1.,0.,0.,0.5),linewidth=0.6,linestyles="dashed")

# ax.set_yscale("log")
ax.set_ylim(0,100)
ax.set_xlim(280,320)
ax.set_title("Usable Spectral Fraction Threshold")
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Transmission [%]')
ax.legend(loc='upper right')

savestr = thesisfig_path + "USF_threshold_illustration.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


plt.show()