import numpy as np
import numpy.ma as ma
from scipy.signal import find_peaks
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as font_manager

# own functions
from .transformations import *
from .transformations import abcd_seriesload
from .transformations import abcd_shuntload
from .transformations import y2abcd
from .transformations import abcd2s
from .utils import res_variance, ABCD_eye

# Edit the font, font size, and axes width
# path = ["./resources/fonts"]
# font_files = font_manager.findSystemFonts(fontpaths=path)
# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)
# mpl.rcParams['font.family'] = 'CMU Serif'
# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['axes.unicode_minus'] = False

# uncomment when rendering for thesis
# mpl.rcParams['text.usetex'] = True

# Font sizes for plot
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 12

mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['xtick.minor.width'] = 0.4
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.minor.width'] = 0.4
mpl.rcParams['ytick.major.width'] = 0.6

mpl.rcParams['axes.linewidth'] = 0.6
mpl.rcParams['lines.linewidth'] = 0.75
mpl.rcParams['lines.markersize'] = 0.75
mpl.rcParams['figure.figsize'] = [2.5,2.5]
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams["savefig.dpi"] = 400

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
        
        assert C < 0, f"Qc value is {Qc}, Ql value is {self.Ql}"
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
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination, sigma_f0=0, sigma_Qc=0) -> None:
        self.f0, self.Ql = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)
        
        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 3, "Z_termination has too many components (max 2 components)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler1 = Coupler(f0=self.f0,Ql=self.Ql,Z_termination=[TransmissionLine.Z0, self.Z_termination[0]],Qi=TransmissionLine.Qi)

        self.Coupler2 = Coupler(f0=self.f0,Ql=self.Ql,Z_termination=[TransmissionLine.Z0, self.Z_termination[-1]],Qi=TransmissionLine.Qi)

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
    def __init__(self, f0, Ql, TransmissionLine : TransmissionLine, Z_termination, sigma_f0=0, sigma_Qc=0) -> None:
        self.f0, self.Ql = res_variance(f0,Ql,TransmissionLine.Qi,sigma_f0,sigma_Qc)

        self.TransmissionLine = TransmissionLine

        assert len(np.atleast_1d(Z_termination)) < 2, "Z_termination has too many components (max 1 component)"
        self.Z_termination = np.atleast_1d(Z_termination)

        self.Coupler = Coupler(f0=self.f0, Ql=self.Ql, Z_termination=[TransmissionLine.Z0, self.Z_termination[0]], Qi=TransmissionLine.Qi, res_length='quarterwave')

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

        
class BaseFilter:
    def __init__(self, f0, Ql, TransmissionLines : dict) -> None:
        self.S_param = None
        self.f = None
        self.S11_absSq = None
        self.S21_absSq = None
        self.S31_absSq = None
        self.S41_absSq = None

        self.Ql_realized = None
        self.f0_realized = None

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

    def S(self,f):
        if np.array_equal(self.f,f):
            return self.S_param
        else:
            S = []
            S.append(abcd2s(self.ABCD(f),self.TransmissionLine_through.Z0))

            for ABCD_to_one_output in self.ABCD_to_MKID(f,ABCD_eye(f)):
                S.append(abcd2s(ABCD_to_one_output,[self.TransmissionLine_through.Z0,self.TransmissionLine_MKID.Z0]))

            self.S_param = S
            self.f = f


            self.S11_absSq = np.abs(self.S_param[0][0][0])**2
            self.S21_absSq = np.abs(self.S_param[0][1][0])**2
            self.S31_absSq = np.abs(self.S_param[1][1][0])**2

            if np.shape(self.S_param)[0] > 2:
                self.S41_absSq = np.abs(self.S_param[2][1][0])**2

            return self.S_param
    
    def plot(self):
        assert self.S_param is not None
        fig, ax =plt.subplots(layout='constrained')

        ax.plot(self.f/1e9,self.S11_absSq,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
        ax.plot(self.f/1e9,self.S21_absSq,label=r'$|S_{21}|^2$',color=(1.,0.,1.))
        ax.plot(self.f/1e9,self.S31_absSq,label=r'$|S_{31}|^2$',color=(0.,0.,0.))

        if self.S41_absSq is not None:
            ax.plot(self.f/1e9,self.S41_absSq,label=r'$|S_{41}|^2$',color=(0.5,0.5,0.5))

        ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
        ax.set_ylabel('Transmission')  # Add a y-label to the axes.
        ax.set_title("Filter response")  # Add a title to the axes.
        ax.legend(loc="lower right");  # Add a legend.
        plt.yscale("log")
        plt.ylim(0.01,1)
        plt.xlim((self.f0-2*self.f0/self.Ql)/1e9,(self.f0+2*self.f0/self.Ql)/1e9)
        # plt.show()
        
    def realized_parameters(self,n_interp=20):
        assert self.S_param is not None

        fq = np.linspace(self.f[0],self.f[-1],n_interp*len(self.f))
 
        S31_absSq_q = np.interp(fq,self.f,self.S31_absSq)

        i_peaks,peak_properties = find_peaks(S31_absSq_q,height=0.5*max(S31_absSq_q),prominence=0.05)

        i_peak = i_peaks[np.argmax(peak_properties["peak_heights"])]
        # f0, as realized in the filter (which is the peak with the highest height given a minimum relative height and prominence)
        self.f0_realized = fq[i_peak]

        # Find FWHM manually:
        HalfMaximum = S31_absSq_q[i_peak] / 2
        diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

        # search window = +/- a number of filter widths
        search_range = [self.f0_realized-3*self.f0/self.Ql, self.f0_realized+3*self.f0/self.Ql]
        
        search_window = np.logical_and(fq > search_range[0],fq < self.f0_realized)
        i_HalfMaximum_lower = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

        search_window = np.logical_and(fq > self.f0_realized,fq < search_range[-1])
        i_HalfMaximum_higher = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

        fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

        self.Ql_realized = self.f0_realized / fwhm



        return self.f0_realized, self.Ql_realized


class ManifoldFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 1.15
        else:
            self.Ql = Ql

        self.Resonator = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

    def ABCD(self, f):
        ABCD = abcd_shuntload(
            Zin_from_abcd(self.Resonator.ABCD(f),self.TransmissionLine_MKID.Z0)
        )
        
        return ABCD
    
    def ABCD_to_MKID(self, f, ABCD_to_termination):
        ABCD_shunt_termination = self.ABCD_shunt_termination(f, ABCD_to_termination)

        ABCD_to_MKID = chain(
            ABCD_shunt_termination,
            self.Resonator.ABCD(f)
        )
        
        return (ABCD_to_MKID,)


class ReflectorFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines: dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 0.75
        else:
            self.Ql = Ql

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        # self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        # Impedance of resonator is equal to onesided connection, due to relfector creating an open condition
        self.Resonator = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0, self.TransmissionLine_MKID.Z0], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

        self.Reflector = Reflector(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = self.TransmissionLine_through.Z0/2, 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

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
        
        return (ABCD_to_MKID,)

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

        return ABCD


class DirectionalFilter(BaseFilter):
    def __init__(self, f0, Ql, TransmissionLines : dict, sigma_f0=0, sigma_Qc=0, compensate=True) -> None:
        super().__init__(f0, Ql, TransmissionLines)

        if compensate == True:
            self.Ql = Ql * 0.95
        else:
            self.Ql = Ql

        self.lmda_quarter = self.TransmissionLine_through.wavelength(f0) / 4
        self.lmda_3quarter = self.TransmissionLine_MKID.wavelength(f0) * 3 / 4
        # self.sep = self.lmda_quarter # quarter lambda is the standard BaseFilter separation

        self.Resonator1 = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )

        self.Resonator2 = Resonator(
            f0 = self.f0, 
            Ql = self.Ql, 
            TransmissionLine = self.TransmissionLine_resonator, 
            Z_termination = [self.TransmissionLine_through.Z0/2, self.TransmissionLine_MKID.Z0/2], 
            sigma_f0 = sigma_f0, 
            sigma_Qc = sigma_Qc
        )
        
    
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
    def __init__(self, FilterClass : BaseFilter, TransmissionLines : dict, f0_min, f0_max, Ql, oversampling=1., sigma_f0=0., sigma_Qc=0., compensate=True) -> None:
        self.S_param = None
        self.f = None
        self.S11_absSq = None
        self.S21_absSq = None
        self.S31_absSq_list = None

        self.f0_realized = None
        self.Ql_realized = None
        self.inband_filter_eff = None
        self.inband_fraction = None

        self.FilterClass = FilterClass
        self.TransmissionLines = TransmissionLines
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.Ql = Ql
        self.sigma_f0 = sigma_f0
        self.sigma_Qc = sigma_Qc
        
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
            self.Filters[i] = FilterClass(f0=self.f0[i], Ql=Ql, TransmissionLines = TransmissionLines, sigma_f0=sigma_f0, sigma_Qc=sigma_Qc, compensate=compensate)
    
    
    def S(self,f):
        if np.array_equal(self.f,f):
            return self.S_param

        else:
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

            self.S_param = S
            self.f = f

            self.S11_absSq = np.abs(self.S_param[0][0][0])**2
            self.S21_absSq = np.abs(self.S_param[0][1][0])**2
            self.S31_absSq_list = []
            
            if np.shape(self.S_param)[0]//self.n_filters == 2:
                for i in np.arange(1,np.shape(self.S_param)[0],2):
                    S_filt1 = np.abs(self.S_param[i][1][0])**2
                    S_filt2 = np.abs(self.S_param[i+1][1][0])**2
                    
                    self.S31_absSq_list.append(S_filt1 + S_filt2)
            else:
                for i in np.arange(1,np.shape(self.S_param)[0],1):
                    S31_absSq = np.abs(self.S_param[i][1][0])**2

                    self.S31_absSq_list.append(S31_absSq)


            return self.S_param
    
    def plot(self):
        assert self.S_param is not None
        fig, ax =plt.subplots(figsize=(5.5,1.5),layout='constrained')

        cmap = cm.get_cmap('rainbow').copy()
        norm = mpl.colors.Normalize(vmin=0, vmax=np.shape(self.S31_absSq_list)[0])

        for i,S31_absSq in enumerate(self.S31_absSq_list):
            ax.plot(self.f/1e9,S31_absSq*100,color=cmap(norm(i)))

        envelope = np.array(self.S31_absSq_list).max(axis=0)
        sum_filters = np.sum(self.S31_absSq_list,axis=0)
        # ax.plot(self.f/1e9,envelope*100,label=r'$\eta_{env}$',color=(0.,0.,0.))
        ax.plot(self.f/1e9,sum_filters*100,label=r'$\sum_3^{N+2}|S_{i1}|^2$',color="0.5")
        ax.plot(self.f/1e9,self.S11_absSq*100,label=r'$|S_{11}|^2$',color=(0.,1.,1.))
        ax.plot(self.f/1e9,self.S21_absSq*100,label=r'$|S_{21}|^2$',color=(1.,0.,1.))

        
        ax.set_xlabel('frequency [GHz]')  # Add an x-label to the axes.
        ax.set_ylabel('Transmission [%]')  # Add a y-label to the axes.
        ax.set_title("Filter response")  # Add a title to the axes.
        ax.legend(loc='lower left');  # Add a legend.
        
        # plt.yscale("log")
        plt.ylim(0,100)
        plt.xlim(np.min(self.f/1e9),np.max(self.f/1e9))

        # plt.show()

    def realized_parameters(self,n_interp=20):
        assert self.S_param is not None

        fq = np.linspace(self.f[0],self.f[-1],n_interp*len(self.f))
        dfq = fq[1] - fq[0]
        self.f0_realized = np.zeros(self.n_filters)
        self.Ql_realized = np.zeros(self.n_filters)
        self.inband_filter_eff = np.zeros(self.n_filters)
        self.inband_fraction = np.zeros(self.n_filters)

        for i in np.arange(self.n_filters):
            if n_interp > 1:
                S31_absSq_q = np.interp(fq,self.f,self.S31_absSq_list[i])
            else:
                S31_absSq_q = self.S31_absSq_list[i]

            n_tries = 5
            width_in_samples = int(np.ceil(self.f0[i] / self.Ql / dfq))
            max_response = np.max(S31_absSq_q)
            for count,prom in enumerate(np.linspace(0.4,0.1,n_tries)):
                try:
                    # prominence is always tricky, add try except loop to gradually decrease prominence in case of no peaks
                    # i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples,prominence=prom*max_response,wlen=10*width_in_samples)
                    i_peaks,_ = find_peaks(S31_absSq_q,height=0.5*max_response,distance=width_in_samples)

                    i_peak = i_peaks[np.argmin(np.abs(fq[i_peaks]-self.f0[i]))]
                except ValueError:
                    if count >= (n_tries-1):
                        # plt.plot(S31_absSq_q)
                        # plt.plot(i_peaks,S31_absSq_q[i_peaks],"x")
                        # plt.show()
                        # print(i)
                        # plt.plot(self.S31_absSq_list[i])
                        raise Exception(f'Fitting peaks could not find a filter response peak with prom = {prom}')
                else:
                    break
                

            
            # f0, as realized in the filterbank (which is the peak with the highest height given a minimum relative height and prominence)
            self.f0_realized[i] = fq[i_peak]

            # Find FWHM manually:
            HalfMaximum = S31_absSq_q[i_peak] / 2
            diff_from_HalfMaximum = np.abs(S31_absSq_q-HalfMaximum)

            # search window = +/- a number of filter widths
            search_range = [self.f0_realized[i]-3*self.f0[i]/self.Ql, self.f0_realized[i]+3*self.f0[i]/self.Ql]
            
            search_window = np.logical_and(fq > search_range[0],fq < self.f0_realized[i])
            i_HalfMaximum_lower = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

            search_window = np.logical_and(fq > self.f0_realized[i],fq < search_range[-1])
            i_HalfMaximum_higher = ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

            fwhm = fq[i_HalfMaximum_higher] - fq[i_HalfMaximum_lower]

            self.Ql_realized[i] = self.f0_realized[i] / fwhm

            # inband_filter_eff
            # inband_fraction
            i_f_max_fb = np.argmin(np.abs(fq-self.f0_realized[0]*1.01))


            self.inband_filter_eff[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / (i_HalfMaximum_higher+1-i_HalfMaximum_lower)
            self.inband_fraction[i] = np.sum(S31_absSq_q[i_HalfMaximum_lower:i_HalfMaximum_higher+1]) / np.sum(S31_absSq_q[:i_f_max_fb])

        return self.f0_realized, self.Ql_realized, self.inband_filter_eff, self.inband_fraction
    
    def reset_and_shuffle(self):
        for i in np.arange(self.n_filters):
            self.Filters[i] = self.FilterClass(f0=self.f0[i], Ql=self.Ql, TransmissionLines = self.TransmissionLines, sigma_f0=self.sigma_f0, sigma_Qc=self.sigma_Qc)

        self.S_param = None
        self.f = None
        self.S11_absSq = None
        self.S21_absSq = None
        self.S31_absSq_list = None

        self.f0_realized = None
        self.Ql_realized = None

        
    


