import numpy as np

def res_variance(f0,Ql,sigma_f0,sigma_Ql):
    # Ql_threshold = Ql - 1.5 * Ql * sigma_Ql
    df = f0 / Ql
    
    f0_var = np.random.normal(f0,df*sigma_f0)
    Ql_var = np.random.normal(Ql,Ql*sigma_Ql)
    return f0_var, Ql_var


def ABCD_eye(f):
    return np.repeat(np.identity(2)[:,:,np.newaxis],len(f),axis=-1)


