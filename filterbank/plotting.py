import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_filterbank(figure,figsize=(10,5),):
    fig, ax =plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(15,6),layout='constrained')

    S31_all = FB.S31_absSq_list


    #ax
    cmap = mpl.cm.get_cmap('rainbow').copy()
    norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(S31_all)[0])

    for i,S31_absSq in enumerate(S31_all):
        
        ax.plot(f/1e9,100*S31_absSq,color=cmap(norm(i)))
        

    # for i,S31_absSq in enumerate(S31_all):
    #     if i in (0,50,125,210,310):
    #         ax.plot(f/1e9,100*S31_absSq,color="0.0",linewidth=2)    

    sum_filters = np.sum(S31_all,axis=0)
    ax.plot(f/1e9,100*sum_filters,label='sum filters',color="0.2")

    envelope = np.array(FB.S31_absSq_list).max(axis=0)
    # ax.plot(f/1e9,100*envelope["DirectionalFilter"],label='envelope',color=(0.,0.,0.))

    # ax.plot(f/1e9,100*FB.S11_absSq,label='S11',color=(0.,1.,1.))
    # ax.plot(f/1e9,100*FB.S21_absSq,label='S21',color=(1.,0.,1.))

    ax.set_ylim(0,100)
    ax.set_xlim(210,450)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Transmission [%]')
    # ax.legend()  # Add a legend.
