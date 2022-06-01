from os import remove
import numpy as np


def chain(ABCDmatrix1,*ABCDmatrices):
    if len(ABCDmatrices) < 1:
        raise RuntimeError("chain() needs at least two ABCD matrices")

    ABCD_out = ABCDmatrix1
    for ABCDmatrix in ABCDmatrices:
        ABCD_out = (ABCD_out.T @ ABCDmatrix.T).T
    return ABCD_out

def unchain(ABCDmatrix,*ABCDmatrices_to_remove,remove_from='front'):
    assert remove_from in ('front','back')

    ABCD_out = ABCDmatrix
    if remove_from == 'front':
        for ABCDmatrix_to_remove in ABCDmatrices_to_remove:
            ABCD_out = (np.linalg.inv(ABCDmatrix_to_remove.T) @ ABCD_out.T).T
    elif remove_from == 'back':
        for ABCDmatrix_to_remove in ABCDmatrices_to_remove:
            ABCD_out = (ABCD_out.T @ np.linalg.inv(ABCDmatrix_to_remove.T)).T
    
    return ABCD_out


def abcd_parallel(ABCD1,ABCD2):
    Y1 = abcd2y(ABCD1)
    Y2 = abcd2y(ABCD2)
    ABCD = y2abcd(Y1 + Y2)

    return ABCD


def abcd_shuntload(Z):
    assert np.array(Z).ndim < 2 
    Z = np.array(Z,dtype=np.cfloat)

    A = np.ones_like(Z)
    B = np.zeros_like(Z)
    C = 1 / Z
    D = np.ones_like(Z)

    ABCD = [[A,B],[C,D]]

    return np.array(ABCD)


def abcd_seriesload(Z):
    assert np.array(Z).ndim < 2 
    Z = np.array(Z,dtype=np.cfloat)

    A = np.ones_like(Z)
    B = Z
    C = np.zeros_like(Z)
    D = np.ones_like(Z)

    ABCD = [[A,B],[C,D]]
    
    return np.array(ABCD)


def Zin_from_abcd(ABCD,Z_L,load_pos='load'):
    assert load_pos in ('load','source')
    
    Z = abcd2z(ABCD)

    Z11 = Z[0][0]
    Z12 = Z[0][1]
    Z21 = Z[1][0]
    Z22 = Z[1][1]

    if load_pos == 'load':
        Z_in = Z11 - Z12 * Z21 / (Z22 + Z_L)
    elif load_pos == 'source':
        Z_in = Z22 - Z12 * Z21 / (Z11 + Z_L)

    return np.array(Z_in)


def abcd2s(ABCD,Z0):
    S = np.empty_like(ABCD,dtype=np.cfloat)
    
    A = ABCD[0][0]
    B = ABCD[0][1]
    C = ABCD[1][0]
    D = ABCD[1][1]

    Z0_1 = np.atleast_1d(Z0)[0]
    Z0_2 = np.atleast_1d(Z0)[-1]

    den = A * Z0_2 + B + C * Z0_1 * Z0_2 + D * Z0_1
    
    S[0][0] = (A * Z0_2 + B - C * np.conj(Z0_1) * Z0_2 - D * np.conj(Z0_1)) / den
    S[0][1] = (2 * (A * D - B * C) * (np.real(Z0_1) * np.real(Z0_2)) **0.5) / den
    S[1][0] = (2 * (np.real(Z0_1) * np.real(Z0_2)) **0.5) / den
    S[1][1] = (-A * np.conj(Z0_2) + B - C * Z0_1 * np.conj(Z0_2) + D * Z0_1) / den

    return np.array(S)


def abcd2z(ABCD):
    try:
        Z = np.empty_like(ABCD,dtype=np.cfloat)
        
        A = ABCD[0][0]
        B = ABCD[0][1]
        C = ABCD[1][0]
        D = ABCD[1][1]

        Z[0][0] = A / C
        Z[0][1] = (A * D - B * C) / C
        Z[1][0] = 1 / C
        Z[1][1] = D / C

        return np.array(Z)
    except RuntimeWarning:
        raise RuntimeError


def abcd2y(ABCD):
    Y = np.empty_like(ABCD,dtype=np.cfloat)
    
    A = ABCD[0][0]
    B = ABCD[0][1]
    C = ABCD[1][0]
    D = ABCD[1][1]

    Y[0][0] = D / B
    Y[0][1] = (B * C - A * D) / B
    Y[1][0] = -1 / B
    Y[1][1] = A / B

    return np.array(Y)


def s2abcd(S,Z0):
    ABCD = np.empty_like(S,dtype=np.cfloat)
    
    S11 = S[0][0]
    S12 = S[0][1]
    S21 = S[1][0]
    S22 = S[1][1]

    Z0_1 = np.atleast_1d(Z0)[0]
    Z0_2 = np.atleast_1d(Z0)[-1]

    den = 2*S21 * np.sqrt( np.real(Z0_1) * np.real(Z0_2) )
    
    ABCD[0][0] = (( np.conj(Z0_1) + S11 * Z0_1 ) / ( 1 - S22 ) + S12 * S21 * Z0_1 ) / den
    ABCD[0][1] = (( np.conj(Z0_1) + S11 * Z0_1 ) * ( np.conj(Z0_2) + S22 * Z0_2 ) - S12 * S21 * Z0_1 * Z0_2 ) / den
    ABCD[1][0] = (( 1 - S11 ) * ( 1 - S22 ) - S12 * S21 )/ den
    ABCD[1][1] = (( 1 - S11 ) * ( np.conj(Z0_2) + S22 * Z0_2 ) + S12 * S21 * Z0_2 ) / den

    return np.array(ABCD)


def z2abcd():
    pass


def y2abcd(Y):
    ABCD = np.empty_like(Y,dtype=np.cfloat)

    Y11 = Y[0][0]
    Y12 = Y[0][1]
    Y21 = Y[1][0]
    Y22 = Y[1][1]

    ABCD[0][0] = -Y22 / Y21
    ABCD[0][1] = -1 / Y21
    ABCD[1][0] = (Y12 * Y21 - Y11 * Y22) / Y21
    ABCD[1][1] = -Y11 / Y21

    return np.array(ABCD)


