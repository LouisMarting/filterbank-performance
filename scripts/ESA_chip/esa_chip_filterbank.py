import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("MacOSX")
from matplotlib import cm
from cycler import cycler
from sympy import symbols, I, cos, sin, re, im, Abs, lambdify, simplify, expand
from sympy import *
import numpy.polynomial.polynomial as poly

# random
rng = np.random.default_rng()

from filterbank.components import Filterbank, TransmissionLine, ManifoldFilter, DirectionalFilter, BaseFilter

plt.style.use('~/Repos/louis-style-docs/default.mplstyle')

fig_path = "./figures/"


### Basic filterbank settings
nF = int(5e4)
f = np.linspace(100e9,300e9,nF)

f0_min = 135e9
f0_max = 270e9
Ql = 30

### Transmission lines
Z0_res = 102
eps_eff_res = 30.9
Qi_res = 1120
TL_res = TransmissionLine(Z0_res,eps_eff_res,Qi=Qi_res)

Z0_thru = 80
eps_eff_thru = 29.5
TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

Z0_kid = 65
eps_eff_kid = 28.9
TL_kid = TransmissionLine(Z0_kid,eps_eff_kid)


TransmissionLinesDict = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_kid
}


## Filterbank
FB = Filterbank(
    FilterClass=DirectionalFilter,
    TransmissionLines=TransmissionLinesDict,
    f0_min=f0_min,
    f0_max=f0_max,
    Ql=Ql,
    oversampling=1,
    sigma_f0=0,
    sigma_Qc=0,
    compensate=False
)

# Caculate S-Parameters and realized values (suppress output)
FB.S(f);
FB.realized_parameters();



# plot filterbank
S31_all = FB.S31_absSq_list

cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=FB.n_filters)

# fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(8,4),layout='constrained')

# for i,S31_absSq in enumerate(S31_all.T):
#     ax.plot(f/1e9,S31_absSq,color=cmap(norm(i)))

# ax.plot(f/1e9,S31_all[:,21],color=cmap(norm(0)))

# plt.show()

S11 = FB.S11_absSq
S21 = FB.S21_absSq


sparse_indices = np.arange(0,len(FB.f0),3)


FB.sparse_filterbank(sparse_indices)

# Check downshift of Qc if that explains the freq shift
FB.coupler_variance(Qc_shifted=20)


# Caculate S-Parameters and realized values (suppress output)
FB.S(f);
FB.realized_parameters();

# plot filterbank
S31_all = FB.S31_absSq_list

sum_filters = np.sum(S31_all,axis=1)

fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(8,4),layout='constrained')

cmap = cm.get_cmap('rainbow').copy()
norm = mpl.colors.Normalize(vmin=0, vmax=FB.n_filters)

for i,S31_absSq in enumerate(S31_all.T):
    ax.plot(f/1e9,S31_absSq,color=cmap(norm(i)))

ax.plot(f/1e9,FB.S11_absSq,color="m",linewidth='1')
ax.plot(f/1e9,FB.S21_absSq,color="c",linewidth='1')

ax.plot(f/1e9,sum_filters,color="#808080",linewidth='1')

ax.set_yscale('log')

ax.set_ylim(1e-2,1e0)

esa_fb_sparse = np.column_stack((FB.f,FB.S11_absSq,FB.S21_absSq,FB.S31_absSq_list,sum_filters))


np.savetxt("esa_fb_sparse.csv",esa_fb_sparse,delimiter=',',header="freq [Hz],S11,S21,Si1 (filters),sum_filters(last column)")

# fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(8,4),layout='constrained')

# ax.plot(f/1e9,FB.S21_absSq,color="#808080",linewidth='1')

# ax.set_yscale('log')

# ax.set_ylim(1e-3,1e0)

print(FB.Ql_realized)

plt.show()


