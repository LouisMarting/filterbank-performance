from cmath import inf
from logging import raiseExceptions
import numpy as np

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

        if topology not in ('series','parallel'):
            raise ValueError(f"topology must be either 'series' or 'parallel', your input was '{topology}'")
        self.topology = topology
        
        if res_length not in ('halfwave','quarterwave'):
            raise ValueError(f"res_length must be either 'halfwave' or 'quarterwave', your input was '{res_length}'")
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


class Resonator:
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination) -> None:
        self.f0 = f0
        self.Ql = Ql
        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler1 = Coupler(f0=f0,Ql=Ql,Z_termination=[TransmissionLine.Z0, Z_termination[0]],Qi=TransmissionLine.Qi)

        self.Coupler2 = Coupler(f0=f0,Ql=Ql,Z_termination=[TransmissionLine.Z0, Z_termination[-1]],Qi=TransmissionLine.Qi)

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
        
        kl = np.arctan( (Z1 - Z_Coupler1 - A) / (-1j * (Z1 * A / Zres - Z_Coupler1 * A / Zres - Zres)) )
        kl[kl<0] = kl[kl<0] + np.pi

        lres = np.real(kl / self.TransmissionLine.wavenumber(self.f0))
        return lres











class Filterbank:
    def __init__(self) -> None:
        pass

class BaseFilter:
    def __init__(self) -> None:
        self.test = 1

    def resonatorABCD(self):
        pass


class DirectionalFilter(BaseFilter):
    def __init__(self) -> None:
        super().__init__()
    



