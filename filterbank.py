import numpy as np
import time

# own functions
from transformations import *
from transformations import abcd_seriesload
from transformations import abcd_shuntload
from transformations import y2abcd

### Physical constants ###
mu0 = np.pi*4e-7
eps0 = 8.854187817620e-12
c0 = 1/np.sqrt(eps0*mu0)



class TransmissionLine:
    def __init__(self,Z0,eps_eff,Qi=np.inf) -> None:

        self.Z0 = Z0
        self.eps_eff = eps_eff
        self.Qi = Qi

    def wavelength(self, f):
        lmda = c0 / np.sqrt(self.eps_eff) / f
        return lmda

    def wavenumber(self, f):
        lmda = self.wavelength(f)
        k = 2 * np.pi / lmda * (1 - 1j / (2*self.Qi))
        return k

    def ABCD(self,f,l):
        g = 1j * self.wavenumber(f)

        A = np.cosh(g * l)
        B = self.Z0 * np.sinh(g * l)
        C = np.sinh(g * l) / self.Z0
        D = np.cosh(g * l)

        ABCD = np.array([[A,B],[C,D]])
        return ABCD



class Coupler:
    def __init__(self, f0, Ql, Z_termination, Qi=np.inf, topology='series', res_length='halfwave') -> None:
        self.f0 = f0
        self.Ql = Ql
        self.Qi = Qi
        if np.isinf(Qi):
            self.Qc = 2 * Ql
        else:
            self.Qc = 2 * Qi * Ql / (Qi - Ql)
        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        assert topology in ('series','parallel')
        self.topology = topology
        
        assert res_length in ('halfwave','quarterwave')
        self.res_length = res_length
        
        self.C = self.capacitance()


    def capacitance(self,Qc=None):
        if not Qc:
            Qc = self.Qc
        
        R1, X1 = np.real(self.Z_termination[0]), np.imag(self.Z_termination[0])
        R2, X2 = np.real(self.Z_termination[-1]), np.imag(self.Z_termination[-1])
        
        n_cycles = {'halfwave':1,'quarterwave':2}.get(self.res_length)

        if self.topology == 'series':
            A = 1
            B = 2 * X1 + 2 * X2
            C = X1**2 + 2 * X1 * X2 + X2**2 - n_cycles * Qc / np.pi * 2 * R1 * R2 + (R1 + R2)**2
        elif self.topology == 'parallel':
            A = (R1 + R2)**2 + (X1 + X2)**2 - n_cycles * Qc / np.pi * 2 * R1 * R2
            B = 2 * (R1 * X2 + R2 * X1) * (R1 + R2) + 2 *(X1 * X2 - R1 * R2) * (X1 + X2)
            C = (R1 * X2 + R2 * X1)**2 + (X1 * X2 - R1 * R2)**2
            
        X = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    
        C_coup = -1 / (2 * np.pi * self.f0 * X)
        return C_coup


    def impedance(self,f):
        Z = -1j / (2 * np.pi * f * self.C)
        return Z
    

    def add_variance(self):
        pass

    def ABCD(self,f):
        Z = self.impedance(f)

        ABCD = abcd_seriesload(Z)

        return ABCD
        


class Resonator:
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination) -> None:
        self.f0 = f0
        self.Ql = Ql
        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler1 = Coupler(f0=f0,Ql=Ql,Z_termination=[TransmissionLine.Z0, self.Z_termination[0]],Qi=TransmissionLine.Qi)

        self.Coupler2 = Coupler(f0=f0,Ql=Ql,Z_termination=[TransmissionLine.Z0, self.Z_termination[-1]],Qi=TransmissionLine.Qi)

        self.l_res = self.resonator_length()


    def resonator_length(self):
        Z1 = self.Z_termination[0]
        Z2 = self.Z_termination[-1]
        Zres = self.TransmissionLine.Z0
        
        if Z1 == 0:
            Z_Coupler1 = 0
        else:
            Z_Coupler1 = self.Coupler1.impedance(self.f0)

        if Z2 == 0:
            Z_Coupler2 = 0
        else:
            Z_Coupler2 = self.Coupler2.impedance(self.f0)
        
        A = Z_Coupler2 + Z2
        
        kl = np.array(np.arctan( (Z1 - Z_Coupler1 - A) / (-1j * (Z1 * A / Zres - Z_Coupler1 * A / Zres - Zres)) ))
        kl[kl<0] = kl[kl<0] + np.pi

        lres = np.real(kl / self.TransmissionLine.wavenumber(self.f0))
        return lres

    def ABCD(self,f):
        ABCD = chain(
                        self.Coupler1.ABCD(f), 
                        self.TransmissionLine.ABCD(f,self.l_res),
                        self.Coupler2.ABCD(f)
                    )
        
        return ABCD
        

class BaseFilter:
    pass

    


class DirectionalFilter():
    def __init__(self, f0, Ql, TransmissionLine_resonator : TransmissionLine, TransmissionLine_through : TransmissionLine, TransmissionLine_MKID : TransmissionLine) -> None:
        self.f0 = f0
        self.Ql = Ql
        self.TransmissionLine_resonator = TransmissionLine_resonator
        self.TransmissionLine_through = TransmissionLine_through
        self.TransmissionLine_MKID = TransmissionLine_MKID
        self.lmda_quarter = TransmissionLine_through.wavelength(f0) / 4
        self.lmda_3quarter = TransmissionLine_MKID.wavelength(f0) * 3 / 4

        self.Resonator1 = Resonator(f0=f0, Ql=Ql, TransmissionLine=TransmissionLine_resonator, Z_termination=[TransmissionLine_through.Z0,TransmissionLine_MKID.Z0])
        self.Resonator2 = Resonator(f0=f0, Ql=Ql, TransmissionLine=TransmissionLine_resonator, Z_termination=[TransmissionLine_through.Z0,TransmissionLine_MKID.Z0])
        
    
    def ABCD(self, f):
        ABCD_lower = chain(
                            
                            self.Resonator1.ABCD(f), 
                            abcd_shuntload(self.TransmissionLine_MKID.Z0),
                            self.TransmissionLine_MKID.ABCD(f, l=self.lmda_3quarter),
                            abcd_shuntload(self.TransmissionLine_MKID.Z0),
                            self.Resonator2.ABCD(f)
                        )
        
        Y_lower = abcd2y(ABCD_lower)
        Y_upper = abcd2y(self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter))
        ABCD = y2abcd(Y_lower + Y_upper)

        return ABCD











class Filterbank:
    def __init__(self) -> None:
        pass


