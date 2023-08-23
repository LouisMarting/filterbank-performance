import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ioff()

from filterbank import Resonator,TransmissionLine,DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank,BaseFilter
from transformations import *
from transformations import abcd_shuntload, chain,unchain,abcd2s
from filterbank.analysis import *
from utils import res_variance






fig_path = "./figures/"

nF = int(2e4)
f = np.linspace(200e9,500e9,nF)

nF = int(5e2)
f2 = np.linspace(345e9,355e9,nF)

f0_min = 220e9
f0_max = 440e9
Ql = 500

## Variances
sigma_Ql = 0.2
sigma_f0 = 0.1

f0_var, Ql_var = res_variance(f0_min,Ql,sigma_f0,sigma_Ql)

print(f"f0_var: {f0_var/1e9}\nQl_var: {Ql_var}")