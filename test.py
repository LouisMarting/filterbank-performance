import numpy as np

from filterbank import Resonator
from filterbank import TransmissionLine



nF = int(1e2)
f = np.linspace(199e9,201e9,nF)


f0 = 200e9
Ql = 112

Z0_res = 45
eps_eff_res = 52

Z0_thru = 72.8

TL_res = TransmissionLine(Z0_res,eps_eff_res)

Res = Resonator(f0,Ql,TL_res,Z0_thru/2)

ABCD_res = Res.ABCD(f)

print(ABCD_res[:,:,0])