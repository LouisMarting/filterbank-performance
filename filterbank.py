import numpy as np
import time

# own functions
from transformations import *
from transformations import abcd_seriesload
from transformations import abcd_shuntload
from transformations import y2abcd
from transformations import abcd2s

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
        
        Z_Coupler1 = self.Coupler1.impedance(self.f0)
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


class Reflector:
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination) -> None:
        self.f0 = f0
        self.Ql = Ql
        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 2, "Z_termination has too many components (max 1 component)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler = Coupler(f0=f0, Ql=Ql, Z_termination=[TransmissionLine.Z0, self.Z_termination[0]], Qi=TransmissionLine.Qi, res_length='quarterwave')

        self.l_res = self.resonator_length()


    def resonator_length(self):
        Z1 = self.Z_termination[0]
        Zres = self.TransmissionLine.Z0
        Z_Coupler = self.Coupler.impedance(self.f0)
        
        kl = np.array(np.arctan( (Z1 - Z_Coupler) / (-1j * (Z1 / Zres - Z_Coupler  / Zres - Zres)) ))
        kl[kl<0] = kl[kl<0] + np.pi

        lres = np.real(kl / self.TransmissionLine.wavenumber(self.f0))
        return lres

    def ABCD(self,f):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                chain(
                    self.Coupler.ABCD(f), 
                    self.TransmissionLine.ABCD(f,self.l_res)
                ),
                0
            )
        )

        return ABCD

        
class BaseFilter():
    def __init__(self, f0, Ql, TransmissionLines : dict) -> None:
        self.f0 = f0
        self.Ql = Ql

        assert all(key in ('through','resonator','MKID') for key in TransmissionLines.keys()), "TranmissionLines dict needs at least the keys: ('through','resonator','MKID')"
        self.TransmissionLines = TransmissionLines
        self.TransmissionLine_through : TransmissionLine = self.TransmissionLines['through']
        self.TransmissionLine_resonator : TransmissionLine = self.TransmissionLines['resonator']
        self.TransmissionLine_MKID : TransmissionLine = self.TransmissionLines['MKID']

        self.sep = self.TransmissionLine_through.wavelength(f0) / 4
    
    def ABCD_sep(self, f):
        ABCD = self.TransmissionLine_through.ABCD(f,self.sep)

        return ABCD

    def ABCD_shunt_termination(self, f, ABCD_to_termination):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                chain(
                    self.ABCD_sep(f),
                    ABCD_to_termination
                ),
                self.TransmissionLine_through.Z0
            )
        )

        return ABCD
    
    def ABCD(self, f):
        # In childs: Add code to construct filter
        pass

    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)
        # In childs: Add code to construct to MKID structure
        pass



class ManifoldFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict) -> None:
        super().__init__(f0, Ql, TransmissionLines)


class ReflectorFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        # Impedance of resonator is equal to onesided connection, due to relfector creating an open condition
        self.Resonator = Resonator(f0=f0, Ql=Ql, TransmissionLine=self.TransmissionLine_resonator, Z_termination=[self.TransmissionLine_through.Z0, self.TransmissionLine_MKID.Z0])
        self.Reflector = Reflector(f0=f0, Ql=Ql, TransmissionLine=self.TransmissionLine_resonator, Z_termination=self.TransmissionLine_through.Z0/2)

    def ABCD(self, f):
        ABCD = chain(
            abcd_shuntload(Zin_from_abcd(self.Resonator.ABCD(f),self.TransmissionLine_MKID.Z0)),
            self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
            self.Reflector.ABCD(f)
        )

        return ABCD

    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_to_MKID = chain(
            ABCD_shunt_termination,
            self.Resonator.ABCD(f)
        )
        
        return ABCD_to_MKID

    def ABCD_shunt_termination(self, f, ABCD_to_termination):
        ABCD = abcd_shuntload(
            Zin_from_abcd(
                chain(
                    self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
                    self.Reflector.ABCD(f),
                    self.ABCD_sep(f),
                    ABCD_to_termination
                ),
                self.TransmissionLine_through.Z0
            )
        )

class DirectionalFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines : dict) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        self.lmda_3quarter = self.TransmissionLine_MKID.wavelength(f0) * 3 / 4
        self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        self.Resonator1 = Resonator(f0=f0, Ql=Ql, TransmissionLine=self.TransmissionLine_resonator, Z_termination=[self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2])
        self.Resonator2 = Resonator(f0=f0, Ql=Ql, TransmissionLine=self.TransmissionLine_resonator, Z_termination=[self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2])
        
    
    def ABCD(self, f):
        ABCD_lower = chain(
            self.Resonator1.ABCD(f), 
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter),
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.Resonator2.ABCD(f)
        )
        
        ABCD = abcd_parallel(ABCD_lower,self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter))

        return ABCD


    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_upper = chain(
            self.TransmissionLine_through.ABCD(f, l=self.lmda_quarter),
            ABCD_shunt_termination,
            self.Resonator2.ABCD(f)
        )

        ABCD_lower = chain(
            self.Resonator1.ABCD(f),
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter)
        )

        ABCD_port4 = abcd_parallel(ABCD_upper,ABCD_lower)

        ABCD_upper_port3 = chain(
            ABCD_upper,
            abcd_shuntload(self.TransmissionLine_MKID.Z0),
            self.TransmissionLine_MKID.ABCD(f,self.lmda_3quarter)
        )
        
        ABCD_port3 = abcd_parallel(ABCD_upper_port3,self.Resonator1.ABCD(f))

        ABCDs = (ABCD_port3,ABCD_port4)
        return ABCDs






class Filterbank:
    def __init__(self, FilterClass : BaseFilter, TransmissionLines : dict, f0_min, f0_max, Ql, oversampling=1, sigma_f0=0, sigma_Ql=0) -> None:
        self.FilterClass = FilterClass
        self.TransmissionLines = TransmissionLines
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.Ql = Ql
        self.sigma_f0 = sigma_f0
        self.sigma_Ql = sigma_Ql
        
        assert oversampling > 0
        self.oversampling = oversampling

        self.n_filters = int(np.floor(1 + np.log10(f0_max / f0_min) / np.log10(1 + 1 / (Ql * oversampling))))
        
        f0 = np.zeros(self.n_filters)
        f0[0] = f0_min
        for i in np.arange(1,self.n_filters):
            f0[i] = f0[i-1] + f0[i-1] / (Ql * oversampling)
        self.f0 = np.flip(f0)

        self.Filters = np.empty(self.n_filters,dtype=BaseFilter)
        for i in np.arange(self.n_filters):
            self.Filters[i] = FilterClass(f0=self.f0[i], Ql=Ql, TransmissionLines = TransmissionLines)
    
    
    def S(self,f):
        Z0_thru = self.TransmissionLines['through'].Z0
        Z0_mkid = self.TransmissionLines['MKID'].Z0
        
        ABCD_preceding = np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)
        ABCD_succeeding = np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)

        ABCD_list = np.empty((2,2,len(f),self.n_filters),dtype=np.cfloat)
        ABCD_sep_list = np.empty((2,2,len(f),self.n_filters),dtype=np.cfloat)
        

        # Calculate a full filterbank chain
        for i, Filter in enumerate(self.Filters):
            Filter : BaseFilter # set the expected datatype of Filter
            

            # Eventually, these indexed lists could be replaced by cached versions.
            ABCD_list[:,:,:,i] = Filter.ABCD(f)
            ABCD_sep_list[:,:,:,i] = Filter.ABCD_sep(f)

            ABCD_succeeding = chain(
                ABCD_succeeding,
                ABCD_list[:,:,:,i],
                ABCD_sep_list[:,:,:,i] # Can we use np.insert() for these and do this calc faster outside of this for loop?
            )
        
        
        S = []

        for i,Filter in enumerate(self.Filters):
            Filter : BaseFilter # set the expected datatype of Filter
            
            # Remove the ith filter from the succeeding filters
            ABCD_succeeding = unchain(
                ABCD_succeeding,
                ABCD_list[:,:,:,i],
                ABCD_sep_list[:,:,:,i]
            )

            # Calculate the equivalent ABCD to the ith detector
            ABCD_to_MKID = Filter.ABCD_to_MKID(f,ABCD_succeeding)

            for ABCD_to_one_output in ABCD_to_MKID:
                ABCD_through_filter = chain(
                    ABCD_preceding,
                    ABCD_to_one_output
                )

                S.append(abcd2s(ABCD_through_filter,[Z0_thru,Z0_mkid]))
            
            ABCD_preceding = chain(
                ABCD_preceding,
                ABCD_list[:,:,:,i],
                ABCD_sep_list[:,:,:,i]
            )
        
        S.insert(0, abcd2s(ABCD_preceding,Z0_thru))
        
        return S