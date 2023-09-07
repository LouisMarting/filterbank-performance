import numpy as np

def res_variance(f0,Ql,Qi,sigma_f0,sigma_Qc):
    if np.isinf(Qi):
        Qc = 2 * Ql
    else:
        Qc = 2 * (Ql * Qi)/(Qi - Ql)
    
    df = f0 / Ql
    
    f0_var = np.random.normal(f0,df*sigma_f0)
    try:
        Qc_var = np.random.normal(Qc,Qc*sigma_Qc)
        assert Qc_var > 0, "Qc variance causes <0 Qc value"
    except AssertionError:
        Qc_var = Qc
        raise UserWarning("consider decreasing the variance applied")

    if np.isinf(Qi):
        Ql_var = Qc_var / 2
    else:
        Ql_var = (Qc_var * Qi) / (Qc_var + 2 * Qi)
    
    return f0_var, Ql_var


def ABCD_eye(f):
    return np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)


