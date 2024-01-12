import numpy as np
from scipy.signal import find_peaks
import copy

from ..utils import ABCD_eye
from ..transformations import abcd2s,chain,unchain
from .transmission_line_filters import BaseFilter


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
            self.Filters[i] = FilterClass(f0=self.f0[i], Ql=Ql, TransmissionLines = copy.deepcopy(TransmissionLines), sigma_f0=sigma_f0, sigma_Qc=sigma_Qc, compensate=compensate)
    
    
    def S(self,f):
        if np.array_equal(self.f,f):
            return self.S_param

        else:
            Z0_thru = self.TransmissionLines['through'].Z0
            Z0_mkid = self.TransmissionLines['MKID'].Z0
            
            ABCD_preceding = ABCD_eye(len(f),dtype=np.cfloat)
            ABCD_succeeding = ABCD_eye(len(f),dtype=np.cfloat)

            ABCD_list = ABCD_eye((len(f),self.n_filters),dtype=np.cfloat)
            ABCD_sep_list = ABCD_eye((len(f),self.n_filters),dtype=np.cfloat)
            

            # Calculate a full filterbank chain
            for i, Filter in enumerate(self.Filters):
                Filter : BaseFilter # set the expected datatype of Filter
                

                # Eventually, these indexed lists could be replaced by cached versions.
                ABCD_list[:,i,:,:] = Filter.ABCD(f)
                ABCD_sep_list[:,i,:,:] = Filter.ABCD_sep(f)

                # Can we use np.insert() for these and do this calc faster outside of this for loop?
                ABCD_succeeding = chain(
                    ABCD_succeeding,
                    ABCD_list[:,i,:,:],
                    ABCD_sep_list[:,i,:,:],
                )
            
            s_parameter_array_size = Filter.n_outputs() * self.n_filters + 2
            S = np.empty((len(f),s_parameter_array_size,),dtype=np.cfloat)

            for i,Filter in enumerate(self.Filters):
                Filter : BaseFilter # set the expected datatype of Filter
                
                # Remove the ith filter from the succeeding filters
                ABCD_succeeding = unchain(
                    ABCD_succeeding,
                    ABCD_list[:,i,:,:],
                    ABCD_sep_list[:,i,:,:]
                )

                # Calculate the equivalent ABCD to the ith detector
                ABCD_to_MKID = Filter.ABCD_to_MKID(f,ABCD_succeeding)

                assert len(ABCD_to_MKID) == Filter.n_outputs(), "Something seriously wrong here"

                for j,ABCD_to_one_output in enumerate(ABCD_to_MKID):
                    ABCD_through_filter = chain(
                        ABCD_preceding,
                        ABCD_to_one_output
                    )
                    S_one_output = abcd2s(ABCD_through_filter,[Z0_thru,Z0_mkid])

                    index = (Filter.n_outputs() * i)+j+2
                    
                    S[:,index] = S_one_output[:,1,0] # Si1
                
                ABCD_preceding = chain(
                    ABCD_preceding,
                    ABCD_list[:,i,:,:],
                    ABCD_sep_list[:,i,:,:]
                )
            
            S_full_FB = abcd2s(ABCD_preceding,Z0_thru)
            S[:,0] = S_full_FB[:,0,0] # S11
            S[:,1] = S_full_FB[:,1,0] # S21

            self.S_param = S
            self.f = f

            self.S11_absSq = np.abs(S[:,0])**2
            self.S21_absSq = np.abs(S[:,1])**2
            if self.Filters[0].n_outputs() == 2:
                self.S31_absSq_list = np.abs(S[:,2::2])**2 + np.abs(S[:,3::2])**2
            else:
                self.S31_absSq_list = np.abs(S[:,2:])**2

            return self.S_param
    

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
                S31_absSq_q = np.interp(fq,self.f,self.S31_absSq_list[:,i])
            else:
                S31_absSq_q = self.S31_absSq_list[:,i]

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
            i_HalfMaximum_lower = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

            search_window = np.logical_and(fq > self.f0_realized[i],fq < search_range[-1])
            i_HalfMaximum_higher = np.ma.masked_array(diff_from_HalfMaximum,mask=~search_window).argmin()

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
