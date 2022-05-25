import numpy as np

from filterbank import Resonator,TransmissionLine,DirectionalFilter
from transformations import chain,unchain



nF = int(1e2)
f = np.linspace(199e9,201e9,nF)


f0 = 200e9
Ql = 112

Z0_res = 22.2
eps_eff_res = 70.7

Z0_thru = 72.8
eps_eff_thru = 42.9

TL_thru = TransmissionLine(Z0_thru,eps_eff_thru)

TL_res = TransmissionLine(Z0_res,eps_eff_res)

TransmissionLines = {
    'through' : TL_thru,
    'resonator' : TL_res,
    'MKID' : TL_thru
}

Res = Resonator(f0,Ql,TL_res,Z0_thru/2)

ABCD_res = Res.ABCD(f)

print(ABCD_res[:,:,0])

TestDirFilter =DirectionalFilter(f0,Ql,TL_res,TL_thru,TL_thru)

ABCD_dir = TestDirFilter.ABCD(f)

ABCD_chain = chain(ABCD_dir,ABCD_dir)

ABCD_unchained = unchain(ABCD_chain,ABCD_dir)

print(np.isclose(ABCD_unchained,ABCD_dir))