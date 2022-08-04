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

        f0_realized[i],Ql_realized[i] = Filterbank.realized_parameters()

    f0_realized = np.array(f0_realized).ravel()
    Ql_realized = np.array(Ql_realized).ravel()
    
    df_variance = Ql * (f0_realized - np.tile(f0,n_filterbanks)) / np.tile(f0,n_filterbanks)
    Ql_variance = Ql_realized / Ql
    
    return f0_realized, Ql_realized, df_variance, Ql_variance
    