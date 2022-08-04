import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ioff()

from filterbank import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from transformations import *
from transformations import abcd_shuntload, chain,unchain,abcd2s
from analysis import *

thesisfig_path = "H:/My Documents/Thesis/Figures/Thesis/Report figures/"

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





for Filter in (ManifoldFilter,ReflectorFilter,DirectionalFilter):

    isolated_filter : BaseFilter = Filter(f0=300e9,Ql=Ql,TransmissionLines=TransmissionLinesDict)

    isolated_filter.S(f2)
    isolated_filter.plot()

    savestr = thesisfig_path + str(Filter.__name__) + "_isolated_filter.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.FigureManagerBase.set_window_title(savestr)
    # plt.close()

    filterbank = Filterbank(
        FilterClass=Filter,
        TransmissionLines=TransmissionLinesDict,
        f0_min=f0_min,
        f0_max=f0_max,
        Ql=Ql
    )

    filterbank.S(f)
    filterbank.plot()

    savestr = thesisfig_path + str(Filter.__name__) + "_filterbank.pdf"
    plt.savefig(fname=savestr)
    fig = plt.gcf()
    fig.FigureManagerBase.set_window_title(savestr)

    # plt.close()






#=============================================
var_settings = [(0.2,0.1)]#[(0,0), (0.1,0.05), (0.2,0.1), (0.3,0.3)]
for var_setting in var_settings:
    OmegaFilterbank = Filterbank(DirectionalFilter,TransmissionLinesDict,f0_min=f0_min,f0_max=f0_max,Ql=Ql, sigma_f0=var_setting[0],sigma_Ql=var_setting[1])

    # OmegaFilterbank.plot()
    f0_realized, Ql_realized, df_variance, Ql_variance = analyse(OmegaFilterbank,f,n_filterbanks=5)
    
    fig, ax =plt.subplots(layout='constrained')

    ax.scatter(f0_realized/1e9,Ql_realized,color=(0.,0.,0.))

    ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
    ax.set_ylabel('realized Ql')  # Add a y-label to the axes.
    ax.set_title("Realized filter parameters")  # Add a title to the axes.
    # ax.legend();  # Add a legend.
    # plt.ylim(-30,0)
    # plt.xlim((self.f0-2*self.f0/self.Ql)/1e9,(self.f0+2*self.f0/self.Ql)/1e9)
    
    savestr = thesisfig_path + f"{OmegaFilterbank.FilterClass.__name__}_Ql_realized_dQ_{var_setting[1]}_df_{var_setting[0]}.pdf"
    plt.savefig(fname=savestr)
    fig.FigureManagerBase.set_window_title(savestr)



    fig, (ax1, ax2) =plt.subplots(1,2,figsize=(5.3,2.5),layout='constrained')
    ax1 : plt.Axes
    ax2 : plt.Axes

    ax1.hist(df_variance,bins=30)

    ax1.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
    ax1.set_title("Realized filter parameters")  # Add a title to the axes.


    ax2.hist(Ql_variance,bins=30)

    ax2.set_xlabel('realized Ql')  # Add an x-label to the axes.
    ax2.set_title("Realized filter parameters")  # Add a title to the axes.


    # plt.ylim(-30,0)
    # plt.xlim((self.f0-2*self.f0/self.Ql)/1e9,(self.f0+2*self.f0/self.Ql)/1e9)
    savestr = thesisfig_path + f"{OmegaFilterbank.FilterClass.__name__}_histogram_dQ_{var_setting[1]}_df_{var_setting[0]}.pdf"
    plt.savefig(fname=savestr)
    fig.FigureManagerBase.set_window_title(savestr)

plt.show()