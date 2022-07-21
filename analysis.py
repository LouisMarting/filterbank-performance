import numpy as np
from scipy import signal
import matplotlib as mpl
from matplotlib import pyplot as plt
import time

from filterbank import Filterbank



def analyse(Filterbank : Filterbank,f,n_filterbanks=1):
    f0 = Filterbank.f0
    Ql = Filterbank.Ql

    f0_realized = [[]] * n_filterbanks
    Ql_realized = [[]] * n_filterbanks

    for i in range(n_filterbanks):
        Filterbank.reset_and_shuffle()
        Filterbank.S(f)
        # Filterbank.plot()

        f0_realized[i],Ql_realized[i] = Filterbank.realized_parameters()

    Filterbank.plot()

    f0_realized = np.array(f0_realized).ravel()
    Ql_realized = np.array(Ql_realized).ravel()
    
    fig, ax =plt.subplots(figsize=(8,6),layout='constrained')

    ax.scatter(f0_realized/1e9,Ql_realized,color=(0.,0.,0.))

    ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
    ax.set_ylabel('realized Ql')  # Add a y-label to the axes.
    ax.set_title("Realized filter parameters")  # Add a title to the axes.
    # ax.legend();  # Add a legend.
    # plt.ylim(-30,0)
    # plt.xlim((self.f0-2*self.f0/self.Ql)/1e9,(self.f0+2*self.f0/self.Ql)/1e9)
    plt.show()


    df_variance = Ql * (f0_realized - np.tile(f0,n_filterbanks)) / np.tile(f0,n_filterbanks)
    Ql_variance = Ql_realized / Ql

    fig, (ax1, ax2) =plt.subplots(1,2,figsize=(16,6),layout='constrained')
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
    plt.show()

    