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

thesisfig_path = "H:/My Documents/Thesis/Presentation/Python/"
# thesisfig_path = "W:/student-homes/m/lmarting/My Documents/Thesis/Presentation/Python/"

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

inband_filter_eff = dict()
f0_compensated_data = dict()

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

fig, (ax1, ax2, ax3) =plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(7,5),layout='constrained')



#ax1
filterbank_model = Filterbank(
    FilterClass=ManifoldFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    # sigma_f0=0.14,
    # sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["ManifoldFilter"],_,inband_filter_eff["ManifoldFilter"],_ = filterbank_model.realized_parameters()

cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(filterbank_model.S31_absSq_list)[0])

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    line, = ax1.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

line.set_label(r'$|S_{i1}|^2$')

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax1.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax1.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax1.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax1.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax1.set_ylim(0,100)
ax1.legend(loc='upper right')

#ax2
filterbank_model = Filterbank(
    FilterClass=ReflectorFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    # sigma_f0=0.14,
    # sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["ReflectorFilter"],_,inband_filter_eff["ReflectorFilter"],_ = filterbank_model.realized_parameters()

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    ax2.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax2.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax2.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax2.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax2.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax2.set_ylim(0,100)

#ax3
filterbank_model = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    # sigma_f0=0.14,
    # sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["DirectionalFilter"],_,inband_filter_eff["DirectionalFilter"],_ = filterbank_model.realized_parameters()

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    ax3.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax3.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax3.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax3.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax3.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax3.set_ylim(0,100)
ax3.set_xlabel('Frequency [GHz]')
#---

# fig.supxlabel('Frequency [GHz]')  # Add an x-label to the axes.
fig.supylabel('Transmission [%]')  # Add a y-label to the axes.
plt.suptitle("Filterbank Model")  # Add a title to the axes.
# ax1.legend();  # Add a legend.
# plt.yscale("log")
# plt.ylim(1,100)
plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = thesisfig_path + "Filterbanks_envelope.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')
for Filter in Filters:
    filt_name = str(Filter.__name__)
    # ax.plot(f/1e9,envelope[filt_name]*100,color=Filtertype_color_dict[filt_name],alpha=0.3)
    ax.scatter(f0_compensated_data[filt_name]/1e9,inband_filter_eff[filt_name]*100,label=legend_filter_name[filt_name],color=Filtertype_color_dict[filt_name])
    

ax.set_ylim(0,100)
ax.set_xlim(195,405)
ax.set_xlabel('Frequency [GHz]')  # Add an x-label to the axes.
ax.set_ylabel('Efficiency [%]')  # Add a y-label to the axes.
ax.set_title("Envelope Efficiency")  # Add a title to the axes.
ax.legend();  # Add a legend.

savestr = thesisfig_path + "Inband_filter_eff.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)




filterbank_variance = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    sigma_f0=0.2,
    sigma_Ql=0.1
)

## Filterbank USF threshold plot
_, _, _, _, avg_envelope_eff, usable_spectral_fraction = analyse_variance(filterbank_variance,f,n_filterbanks=1)

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(3,2.5),layout='constrained')

#ax1
cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=20, vmax=35)

for i,S31_absSq in enumerate(filterbank_variance.S31_absSq_list):
    ax.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))



envelope = np.array(filterbank_variance.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_variance.S31_absSq_list,axis=0)
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


fig, (ax1, ax2, ax3) =plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(7,5),layout='constrained')



#ax1
filterbank_model = Filterbank(
    FilterClass=ManifoldFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    sigma_f0=0.14,
    sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["ManifoldFilter"],_,inband_filter_eff["ManifoldFilter"],_ = filterbank_model.realized_parameters()

cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(filterbank_model.S31_absSq_list)[0])

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    line, = ax1.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

line.set_label(r'$|S_{i1}|^2$')

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax1.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax1.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax1.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax1.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax1.set_ylim(0,100)
ax1.legend(loc='upper right')

#ax2
filterbank_model = Filterbank(
    FilterClass=ReflectorFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    sigma_f0=0.14,
    sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["ReflectorFilter"],_,inband_filter_eff["ReflectorFilter"],_ = filterbank_model.realized_parameters()

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    ax2.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax2.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax2.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax2.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax2.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax2.set_ylim(0,100)

#ax3
filterbank_model = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    sigma_f0=0.14,
    sigma_Ql=0.07
)

filterbank_model.S(f)

f0_compensated_data["DirectionalFilter"],_,inband_filter_eff["DirectionalFilter"],_ = filterbank_model.realized_parameters()

for i,S31_absSq in enumerate(filterbank_model.S31_absSq_list):
    ax3.plot(f/1e9,S31_absSq*100,color=cmap(norm(i)))

envelope = np.array(filterbank_model.S31_absSq_list).max(axis=0)
sum_filters = np.sum(filterbank_model.S31_absSq_list,axis=0)
ax3.plot(f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.),linestyle=":")
ax3.plot(f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
ax3.plot(f/1e9,filterbank_model.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
ax3.plot(f/1e9,filterbank_model.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))


ax3.set_ylim(0,100)
ax3.set_xlabel('Frequency [GHz]')
#---

# fig.supxlabel('Frequency [GHz]')  # Add an x-label to the axes.
fig.supylabel('Transmission [%]')  # Add a y-label to the axes.
plt.suptitle("Filterbank Model")  # Add a title to the axes.
# ax1.legend();  # Add a legend.
# plt.yscale("log")
# plt.ylim(1,100)
plt.xlim(np.min(f)/1e9,np.max(f)/1e9)

savestr = thesisfig_path + "Filterbanks_envelope_variance.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)


data_run965 = np.genfromtxt("data/Spectrum.run965.all.csv",delimiter=",")

data_run966 = np.genfromtxt("data/Spectrum.run966.all.csv",delimiter=",")

plot_measurements_presentation(data_run966)

fig = plt.gcf()
savestr = thesisfig_path + f"Filter_measurement_presentation.pdf"
plt.savefig(fname=savestr)
fig.canvas.manager.set_window_title(savestr)

plt.show()


