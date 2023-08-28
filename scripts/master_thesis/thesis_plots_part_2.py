import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager

plt.ioff()

from filterbank.components import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from filterbank.transformations import *
from filterbank.transformations import abcd_shuntload, chain,unchain,abcd2s
from filterbank.analysis import *

# thesisfig_path = "H:/My Documents/Thesis/Figures/Thesis/Report figures/"
thesisfig_path = "W:/student-homes/m/lmarting/My Documents/Thesis/Figures/Thesis/Report figures/"

# Edit the font, font size, and axes width
path = ["./resources/fonts"]
font_files = font_manager.findSystemFonts(fontpaths=path)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
mpl.rcParams['font.family'] = 'CMU Serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
# mpl.rcParams['mathtext.fontset'] = 'custom'

# uncomment when rendering for thesis
# mpl.rcParams['text.usetex'] = True

# Font sizes for plot
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 12

mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.minor.width'] = 0.4
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.minor.width'] = 0.4
mpl.rcParams['ytick.major.width'] = 0.6

mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['lines.markersize'] = 0.75
mpl.rcParams['figure.figsize'] = [2.5,2.5]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams["savefig.dpi"] = 400

##########################################################
## Start of plotting

colors = [
    np.array([90,136,237])/255,
    np.array([86,199,74])/255,
    np.array([242,131,45])/255,
    np.array([132,82,201])/255,
    np.array([108,98,96])/255
]

data_run965 = np.genfromtxt("data/Spectrum.run965.all.csv",delimiter=",")

data_run966 = np.genfromtxt("data/Spectrum.run966.all.csv",delimiter=",")

plot_measurements_filter(data_run965,data_run966)

fig = plt.gcf()
savestr = thesisfig_path + f"Filter_measurement_filter_response.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)

plot_measurements(data_run966)

fig = plt.gcf()
savestr = thesisfig_path + f"Filter_measurement_raw_data.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




## Simulation data comparison
data_simulations_fb = np.genfromtxt("data/experimental_filterbank_3_filters.csv", delimiter=",")

n_filters = 3
nF_sim = 12001
f = data_simulations_fb[:nF_sim,0]

f0_design = np.array([354.7, 351.2, 347.7])

f0_min = np.min(f0_design[:n_filters]) * 1e9
f0_max = np.min(f0_design[0]) * 1e9
Ql = 100 #116.5

Z0_res = 48.1
eps_eff_res = 52.63

Z0_thru = 50.3
eps_eff_thru = 32

TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

TL_res = TransmissionLine(Z0_res,eps_eff_res)

TransmissionLinesDict = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_thru
}

filterbank_model = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    compensate=False
)

# filterbank_model.f0 = f0_design[:n_filters]*1e9
# filterbank_model.n_filters = n_filters
filterbank_model.Ql = 112
filterbank_model.reset_and_shuffle()

filterbank_model.S(f*1e9)

print(f"First filter resonator length: {filterbank_model.Filters[0].Resonator1.l_res}")
print(f"Second filter resonator length: {filterbank_model.Filters[1].Resonator1.l_res}")
print(f"Third filter resonator length: {filterbank_model.Filters[2].Resonator1.l_res}")


fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')


cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=n_filters-1)


#### SIMULATIONS



for i in range(n_filters):
    S31_absSq = np.abs(data_simulations_fb[2*nF_sim+2*i*nF_sim:2*nF_sim+2*i*nF_sim+nF_sim,1])**2 + np.abs(data_simulations_fb[3*nF_sim+2*i*nF_sim:3*nF_sim+2*i*nF_sim+nF_sim,1])**2
    ax.plot(f,S31_absSq,color=colors[i],label=f"Filter {i+1}")

sim_lines = ax.get_lines()

S11_absSq = np.abs(data_simulations_fb[:nF_sim,1])**2
S21_absSq = np.abs(data_simulations_fb[nF_sim:2*nF_sim,1])**2
# ax.plot(f,S11_absSq*100,label='S11',color=(0.,1.,1.),linewidth=0.6)
# ax.plot(f,S21_absSq*100,label='S21',color=(1.,0.,1.),linewidth=0.6)

#### MODEL

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    ax.plot(filterbank_model.f/1e9,S31_absSq*1,color=colors[i],linestyle="--",alpha=0.4)

    n_interp = 20
    fq = np.linspace(f[0],f[-1],n_interp*len(f))
    
    S31_absSq_q = np.interp(fq,f,S31_absSq)

    i_peaks,peak_properties = find_peaks(S31_absSq_q,height=0.5*max(S31_absSq_q),prominence=0.05)

    i_peak = i_peaks[np.argmax(peak_properties["peak_heights"])]
    # f0, as realized in the filter (which is the peak with the highest height given a minimum relative height and prominence)
    f0_realized = fq[i_peak]

    # Find FWHM manually:
    HalfMaximum = S31_absSq_q[i_peak] / 2
    diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

    # search window = +/- a number of filter widths
    search_range = [f0_realized-3*f0_design[i]/Ql, f0_realized+3*f0_design[i]/Ql]

    search_window = np.logical_and(fq > search_range[0],fq < f0_realized)
    i_HalfMaximum_lower = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    search_window = np.logical_and(fq > f0_realized,fq < search_range[-1])
    i_HalfMaximum_higher = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

    fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

    Ql_realized = f0_realized / fwhm

    print(f"f0 realized: {f0_realized} GHz")
    print(f"Ql realized: {Ql_realized}")

# ax.plot(filterbank_model.f/1e9,filterbank_model.S11_absSq*100,label='S11',color=(0.,1.,1.),linestyle="--",alpha=0.4,linewidth=0.6)
# ax.plot(filterbank_model.f/1e9,filterbank_model.S21_absSq*100,label='S21',color=(1.,0.,1.),linestyle="--",alpha=0.4,linewidth=0.6)

ax.set_yscale("log")
ax.set_ylim(0.01,1)
ax.set_xlim(340,365)
ax.set_title("SONNET Simulation")
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Transmission')

solid, = ax.plot([0,0],[0,0],color=[.0,.0,.0],label="Simulation")
dashed, = ax.plot([0,0],[0,0],color=[.0,.0,.0],linestyle="--",label="Model")
type_legend = ax.legend(handles=[solid,dashed],loc="center right")
ax.add_artist(type_legend)

# ax.legend(handles=sim_lines,loc="upper right")

savestr = thesisfig_path + "simulation_vs_model.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


####################### Coupler plots ###################

####### Z0 vs w_res
meas_data = np.genfromtxt("data/Z0_vs_w_res_data_01.csv", delimiter=",")
fit_data = np.genfromtxt("data/Z0_vs_w_res_data_02.csv", delimiter=",")

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(2.5,2),layout='constrained')


ax.plot(fit_data[:,0],fit_data[:,1],label='Best fit',linewidth=1,color=colors[2])
ax.plot(meas_data[:,0],meas_data[:,1],marker='x',linestyle='',ms=3,mew=0.3,alpha=0.9,label='Simulated',color=colors[0])

ax.set_title("Resonator Impedance")
# ax.set_yscale("log")
ax.set_xlabel(r"$w_\mathrm{res}$ [$\mu m$]")
ax.set_ylabel(r"$Z_0$ [$\Omega$]")
ax.legend()

savestr = thesisfig_path + "Z0_vs_w_res.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)



######## Eps_eff vs w_res
meas_data = np.genfromtxt("data/Eps_eff_vs_w_res_data_01.csv", delimiter=",")
fit_data = np.genfromtxt("data/Eps_eff_vs_w_res_data_02.csv", delimiter=",")

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(2.5,2),layout='constrained')


ax.plot(fit_data[:,0],fit_data[:,1],label='Best fit',linewidth=1,color=colors[2])
ax.plot(meas_data[:,0],meas_data[:,1],marker='x',linestyle='',ms=3,mew=0.3,alpha=0.9,label='Simulated',color=colors[0])

ax.set_title("Epsilon Effective")
# ax.set_yscale("log")
ax.set_xlabel(r"$w_\mathrm{res}$ [$\mu m$]")
ax.set_ylabel(r"$\varepsilon_\mathrm{eff}$")
ax.legend()

savestr = thesisfig_path + "Eps_eff_vs_w_res.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


######## Optimal w_res
meas_data = np.genfromtxt("data/w_res_optimal_data_01.csv", delimiter=",")
fit_data = np.genfromtxt("data/w_res_optimal_data_02.csv", delimiter=",")

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(2.5,2),layout='constrained')


ax.plot(fit_data[:,0],fit_data[:,1],label='Best fit',linewidth=1,color=colors[2])
ax.plot(meas_data[:,0],meas_data[:,1],marker='x',linestyle='',ms=3,mew=0.3,alpha=0.9,label='Simulated',color=colors[0])

ax.set_title(r"Optimal $w_\mathrm{res}$")
# ax.set_yscale("log")
ax.set_ylabel(r"$w_\mathrm{res}$ [$\mu m$]")
ax.set_xlabel("Frequency [GHz]")
ax.legend()

savestr = thesisfig_path + "w_res_optimal.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




######## coupling strength vs w_res
thresh_data = np.genfromtxt("data/coup_vs_w_res_data_01.csv", delimiter=",")

cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=2, vmax=66)

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(2.5,2),layout='constrained')

for i in np.arange(2,67):
    meas_data = np.genfromtxt(f"data/coup_vs_w_res_data_{i:02d}.csv", delimiter=",")
    ax.plot(meas_data[:,0],meas_data[:,1],color=cmap(norm(68-i)))


ax.plot(thresh_data[:,0],thresh_data[:,1],linestyle='--',color=[0.,0.,0.],zorder=100)

norm_cbar = mpl.colors.Normalize(vmin=0.5, vmax=3.7)

fig.colorbar(cm.ScalarMappable(norm=norm_cbar,cmap=cmap), ax=ax,label=r"$w_\mathrm{res}$ [$\mu m$]")


ax.set_title("Coupling Strength")
# ax.set_yscale("log")
ax.set_ylabel(r"$|S_\mathrm{ij}|^2$ [dB]")
ax.set_xlabel("Frequency [GHz]")
# ax.legend()

savestr = thesisfig_path + "coup_vs_w_res.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


### Test chip filterbank design
nF = int(5e3)
f = np.linspace(180e9,440e9,nF)

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

filterbank_model = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
)

filter_id = np.array([4,8,12])
filter_id = np.append(filter_id,np.arange(16,53))
filter_id = np.append(filter_id,np.array([56,60,64]))

filterbank_model.f0 = filterbank_model.f0[filter_id]
filterbank_model.n_filters = len(filter_id)
filterbank_model.Filters = filterbank_model.Filters[filter_id]

filterbank_model.reset_and_shuffle()

filterbank_model.S(f)


fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(5.5,1.5),layout='constrained')

S31_all = filterbank_model.S31_absSq_list

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

savestr = thesisfig_path + "Filterbank_test_chip.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




############# Losses plot ##############
losses_settings_colors = [
    np.array([90,136,237])/255,
    np.array([86,199,74])/255,
    np.array([242,131,45])/255,
    np.array([132,82,201])/255,
    np.array([108,98,96])/255
]


nF = int(5e3)
f = np.linspace(180e9,440e9,nF)

f0_min = 200e9
f0_max = 400e9
Ql = 100 

Z0_res = 80
eps_eff_res = 40

Z0_thru = 80
eps_eff_thru = 40

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

losses = [np.inf,500,200]

inband_filter_eff = np.empty((np.size(losses,axis=0),len(Filters),70))


fig, axes =plt.subplots(nrows=len(Filters),ncols=2,sharex=True,sharey=True,figsize=(4,5),layout='constrained')
for losses_i,losses_setting in enumerate(losses):
    for i,row in enumerate(axes):

        TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

        TL_res = TransmissionLine(Z0_res,eps_eff_res,Qi=losses[losses_i])

        TransmissionLinesDict = {
            'through' : TL_thru,
            'resonator' : TL_res,
            'MKID' : TL_thru
        }

        filterbank = Filterbank(
            FilterClass=Filters[i],
            TransmissionLines=TransmissionLinesDict,
            f0_min=f0_min,
            f0_max=f0_max,
            Ql=Ql
        )

        filterbank.S(f)

        f0_realized, Ql_realized, inband_filter_eff[losses_i,i,:], _ = filterbank.realized_parameters()

        df_variance = Ql * (f0_realized - filterbank.f0) / filterbank.f0
        Ql_variance = (Ql_realized-Ql) / Ql

        hist_data = [df_variance,Ql_variance]

        for j,ax in enumerate(row):
            ax.hist(hist_data[j]*100,bins=np.linspace(-100,100,num=41),facecolor=losses_settings_colors[losses_i],alpha=0.4)


custom_lines = [Line2D([0], [0], color=losses_settings_colors[0], lw=4),
                Line2D([0], [0], color=losses_settings_colors[1], lw=4),
                Line2D([0], [0], color=losses_settings_colors[2], lw=4)]

# axes[0,1].legend(custom_lines, [r"$\sigma_{f_i} = 0 \, \% \: \sigma_{Q_\mathrm{L}} = 0\, \%$",r"$\sigma_{f_i} = 12 \, \% \: \sigma_{Q_\mathrm{L}} = 7\, \%$",r"$\sigma_{f_i} = 30 \, \% \: \sigma_{Q_\mathrm{L}} = 15\, \%$"], loc='upper right')
axes[0,1].legend(custom_lines, [r"$Q_\mathrm{i} = \infty$",r"$Q_\mathrm{i} = 500$",r"$Q_\mathrm{i} = 200$"], loc='upper right')

axes[0,0].set_title("Frequency Scatter")
axes[0,1].set_title("Ql Scatter")
axes[-1,0].set_xlabel(r"$(f_i^{measured} - f_i) / \Delta f_i$ [%]")
axes[-1,1].set_xlabel(r"$(Q_{L,i}^{measured} - Q_L) / Q_L$ [%]")
axes[0,0].set_ylabel("Manifold filter")
axes[1,0].set_ylabel("Reflector filter")
axes[2,0].set_ylabel("Directional filter")
axes[0,0].set_xlim(-100,100)
axes[0,1].set_xlim(-100,100)
axes[0,0].set_ylim(0,200)
axes[0,1].set_ylim(0,200)

savestr = thesisfig_path + f"Histogram_losses.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)





## BOXPLOT 

ticks = ["Manifold\nFilter","Reflector\nFilter","Directional\nFilter"]

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')

positions_base = np.array(np.arange(len(ticks)))

positions_offset = np.linspace(-0.25,0.25,len(losses))

for losses_i,loss_setting in enumerate(losses):

    inband_filter_eff_plot = ax.boxplot(
        inband_filter_eff[losses_i].T*100,
        positions = positions_base + positions_offset[losses_i],
        widths= 0.5 / len(losses)
        )
    
    def define_box_properties(plot_name, color_code):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)
            plt.setp(plot_name['fliers'], markeredgecolor=color_code,markersize=4)
    
    define_box_properties(inband_filter_eff_plot,losses_settings_colors[losses_i])



custom_lines = [Line2D([0], [0], color=losses_settings_colors[0], lw=4),
                Line2D([0], [0], color=losses_settings_colors[1], lw=4),
                Line2D([0], [0], color=losses_settings_colors[2], lw=4)]

# axes[0,1].legend(custom_lines, [r"$\sigma_{f_i} = 0 \, \% \: \sigma_{Q_\mathrm{L}} = 0\, \%$",r"$\sigma_{f_i} = 12 \, \% \: \sigma_{Q_\mathrm{L}} = 7\, \%$",r"$\sigma_{f_i} = 30 \, \% \: \sigma_{Q_\mathrm{L}} = 15\, \%$"], loc='upper right')
ax.legend(custom_lines, [r"$Q_\mathrm{i} = \infty$",r"$Q_\mathrm{i} = 500$",r"$Q_\mathrm{i} = 200$"], loc='upper left')

ax.set_xticks(positions_base,ticks)
# ax1.set_ylim(0,100)
ax.yaxis.grid(True)
ax.set_title("In-Band Filter Efficiency")
ax.set_ylabel('Efficiency [%]')
ax.tick_params(axis='x',which='minor',bottom=False,top=False)


savestr = thesisfig_path + "Boxplot_losses.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)



plt.show()