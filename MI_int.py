import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *
from time import time
from multiprocessing import Pool, Manager, Value, Array
from functools import partial
from scipy.integrate import simpson

def f_li_given_S(li, q, sigma):
    """f(li|s) for all s in Zq

    Parameters
    ----------
    li: float
        Leakage of share i
    q: int
        Prime
    sigma: float
        Standard deviation for Gaussian leakages

    Returns
    -------
    f(li|S): array size S
            f(li|S)[i] = f(li|si)
    """
    Zq = np.arange(q)
    hw_set = HW(Zq)
    hw_set_u = np.unique(hw_set)
    li_ = np.repeat(li, repeats=hw_set_u.shape[0], axis=0)
    pdf_hw = pdf_normal(li_, hw_set_u, sigma)
    f_li = pdf_hw[hw_set]
    return f_li

def f_L_given_S(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
    """Compute f(l|s) for all possible value of the secret s and multiple traces
    Only for 2 shares (extendability unknown)
    Parameters
    ----------
    L: 2D array
        L = {l}_n_traces where l = [l1, l0]
    s_range : array size S
        Possible values for s
    prior_s : array size S
        Prior proba of s (binomial distribution)
    shares: array shape: (n_shares-1, q^(n_shares-1))
        Precomputed all possible randomness (l1, ..., ld-2)
    masked_S: array shape: (S, q^(n_shares-1))
        Precomputed masked secret (l0) for all possible value of secret
    n_shares: int
        Number of shares
    q: int
        Prime
    sigma: float
        Standard deviation for Gaussian leakages

    Returns
    -------
    f(L|S): array size n_traces x S
            f(l|S)[i] = f(l|si)
            f(l|si) = sum_s_1 f(l0|s0)f(l1|s1)

    """
    n_traces  = L.shape[1]
    Zq = np.arange(q)
    hw_set = HW(Zq)
    hw_set_u = np.unique(hw_set)
    f_L_given_si = np.ones((n_traces, q))
    for ns in range(n_shares-1):
        Li = np.expand_dims(L[ns], axis=1)
        f_Li = np.apply_along_axis(pdf_normal, axis=1, arr=Li, mu=hw_set_u, sigma=sigma)
        hw_share_i = HW(shares[:, ns])
        f_Li = f_Li[:, hw_share_i]
        f_L_given_si  *= f_Li
    L0 = np.expand_dims(L[n_shares-1], axis=1)
    f_L0 = np.apply_along_axis(pdf_normal, axis=1, arr=L0, mu=hw_set_u, sigma=sigma)
    f_L_given_S_prodacc = []
    for ms in masked_S:
        hw_maskeds = HW(ms)
        f_L0_ms = f_L0[:, hw_maskeds]
        f_L_given_S_prodacc.append(f_L_given_si*f_L0_ms)
    f_L_given_S_prodacc = np.array(f_L_given_S_prodacc)
    f_L_given_S = f_L_given_S_prodacc.sum(axis=2, keepdims=True)
    prior_s_ = prior_s[..., np.newaxis, np.newaxis]
    f_L_given_S_ = f_L_given_S*prior_s_
    return f_L_given_S_/(q**(n_shares -1))

def int_g(L, s_range, shares, masked_S, n_shares, q, sigma):
    fLS = f_L_given_S(L, s_range, shares_sub, masked_S_sub, n_shares, q, sigma)
    n_traces = L.shape[1]
    G = []
    fLS_sum = fLS.sum(axis=0, keepdims=True).squeeze()
    for i, s in enumerate(s_range):
        fLs = fLS[i].squeeze
        psL = fLs/fLS_sum
        g.append(fLs*np.log2(psL))

    return 0


def test_f():
    q = 23
    n_shares = 2
    sigma = 0.03162277660168379
    s_range = np.array([-2, -1, 0, 1, 2])
    n_samples = 4
    prior_s = prior_ps()
    prior_s_ = prior_s[..., np.newaxis, np.newaxis]
    print("prior_s", prior_s.shape, prior_s_)
    s = np.array([s_range[2]])
    shares = gen_shares_am(s, q, n_shares, n_samples)
    leakages = gen_leakages(shares, n_samples, sigma)
    L = np.array([val for val in leakages.values()])
    print("leakages shape", L.shape)
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    pretty_print_dict(shares, tag="shares")
    pretty_print_dict_HW(shares, tag="HW shares")
    pretty_print_dict(leakages, tag="leakages")
    f_S = f_L_given_S(L, s_range, prior_s, shares_sub, masked_S_sub, n_shares, q, sigma)
    print("f_S", f_S.shape, f_S)
    f_S_ = f_S*prior_s_
    print("After prior s", f_S_.shape, f_S_)
    f_S0 = f_S_[0]
    print("f_s=0", f_S0.shape, f_S0)
    sum_f = f_S_.sum(axis=0, keepdims=True)
    print("sum f_L", (sum_f.squeeze()).shape, sum_f)
    psL0 = f_S0.squeeze()/sum_f.squeeze()
    print(psL0)
    f_S2 = f_S_[2].squeeze()
    print("f_s=2", f_S2.shape, f_S2)
    psL2 = f_S2.squeeze()/sum_f.squeeze()
    print("psL2", psL2)
    g = f_S2*np.log2(psL2)
    print("g", g)



if __name__ == '__main__':
    test_f()
    # prior_s = prior_ps()
    # prior_s_ = prior_s[..., np.newaxis, np.newaxis]
    # print("prior_s", prior_s.shape, prior_s_.shape)
    # s_0 = np.array([[ 4.69030088],  [26.60698543],  [25.57159243],  [ 6.44741229]])
    # print(s_0.shape, s_0*0.0625)
