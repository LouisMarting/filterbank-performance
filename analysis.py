import numpy as np
from scipy import signal
import matplotlib as mpl
from matplotlib import pyplot as plt
import time

from filterbank import DirectionalFilter,ReflectorFilter,ManifoldFilter,Filterbank
from utils import get_S11_S21_S31


def analyse(Filterbank : Filterbank,f):
    S = Filterbank.S(f)

    S11_absSq,S21_absSq,S31_absSq = get_S11_S21_S31()