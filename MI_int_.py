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
from tqdm import tqdm
import os
np.seterr(invalid="ignore")

def f_L_given_S(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
    """Compute f(l|s) for all possible value of the secret s and multiple traces
    Only for 2 shares (extendability unknown)
    f(l|s) = sum_s1 p(s1)f(l0|s0)f(l1|s1)

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
            f(L|S)[i] = f(l|S)
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
    fLS = []
    for ms in masked_S:
        hw_maskeds = HW(ms)
        f_L0_ms = f_L0[:, hw_maskeds]
        f_L_given_S_prod = f_L_given_si*f_L0_ms
        f_L_given_S_prod_sum = f_L_given_S_prod.sum(axis=1)
        fLS.append(f_L_given_S_prod_sum)
    fLS_ = np.array(fLS)
    fLS_ = prior_s.squeeze(axis=2)*fLS_
    return fLS_

def int_g(L, G, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
    """Compute G = p(s) f(l|s) log2(p(s|l)) for multiple traces
    for later integral

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
    G: array size L

    """
    fLS_ = f_L_given_S(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
    fLS = fLS_.astype(np.longdouble)
    n_traces = L.shape[1]
    G = np.zeros(n_traces)
    fLS_sum = fLS.sum(axis=0, keepdims=True).squeeze()
    for i, s in enumerate(s_range):
        fLs = fLS[i].squeeze()
        psL = np.where(fLS_sum == 0, 0, fLs/fLS_sum)
        psi = prior_s[i].squeeze()
        g = np.where(psL == 0.0, -0.0, fLs*np.log2(psL))
        G += g
    return G


def int_g_acc(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma, idx, G_acc):
    """Accumulate G = p(s) f(l|s) log2(p(s|l)) for multiple traces to G_acc start from idx
    where G_acc is integral values for all traces

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
    idx: int
        Index of G_acc to accumulate
    G_acc: array
        Accumulate array for later intergral

    Returns
    -------
    0

    """

    fLS_ = f_L_given_S(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
    fLS = fLS_.astype(np.longdouble)
    n_traces = L.shape[1]
    G = np.zeros(n_traces)
    fLS_sum = fLS.sum(axis=0, keepdims=True).squeeze()
    print(f"sigma: {sigma} lmin: {lmin}, lmax: {lmax}, n_points: {n_points}")
    for i, s in enumerate(s_range):
        fLs = fLS[i].squeeze()
        psL = np.where(fLS_sum == 0, 0, fLs/fLS_sum)
        psi = prior_s[i].squeeze()
        g = np.where(psL == 0.0, -0.0, fLs*np.log2(psL))
        G += g
    G_acc[idx]=G
    return 0

def ent_s(prior_s):
    """Entropy for prior proba
    """
    log_2p = np.log2(prior_s)
    return -(prior_s*log_2p).sum()


def MI_int(q, prior_s, G, int_l0, int_l1):
    """Compute MI from precomputed intergral array G
    Using double Simpson's law on int_l0 & int_l1 points
    """
    int_size = int_l0.shape[0]
    G_ = G.reshape(int_size, int_size)
    CE = simpson(simpson(G_, int_l1), int_l0)/q
    return ent_s(prior_s) + CE


def precompute_p(sigma, s_range, prior_s, shares, masked_S, n_shares, q, op, dense=50):
    Zq = np.arange(q)
    hw = HW(Zq)
    maxhw = hw.max()
    lmin = 0 - sigma*5
    lmax = maxhw + sigma*5
    n_points = int(lmax-lmin)*dense
    print(f"Precompute sigma: {sigma} lmin: {lmin}, lmax: {lmax}, n_points: {n_points}")
    int_l0, int_l1, int_L = int_l(lmin, lmax, n_points=n_points)
    L0 = int_L[0].reshape(n_points*n_points)
    L1 = int_L[1].reshape(n_points*n_points)
    L = np.vstack((L1, L0))
    L_chunks = np.split(L, 100, axis=1)
    i_ = 0
    with open(f"G/{q}/{sigma}_{op}_{dense}.npy", "wb") as f:
        np.save(f, int_l0)
        np.save(f, int_l1)
        for L_chunk in L_chunks:
            fLS_ = f_L_given_S(L_chunk, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
            np.save(f, fLS_)
            i_ += 1
def precompute_p_(sigma, s_range, prior_s, shares, masked_S, n_shares, q, op, dense=50):
    Zq = np.arange(q)
    hw = HW(Zq)
    maxhw = hw.max()
    lmin = 0 - sigma*5
    lmax = maxhw + sigma*5
    n_points = int(lmax-lmin)*dense
    print(f"Precompute sigma: {sigma} lmin: {lmin}, lmax: {lmax}, n_points: {n_points}")
    int_l0, int_l1, int_L = int_l(lmin, lmax, n_points=n_points)
    L0 = int_L[0].reshape(n_points*n_points)
    L1 = int_L[1].reshape(n_points*n_points)
    L = np.vstack((L1, L0))
    L_chunks = np.split(L, 500, axis=1)
    i_ = 0
    with open(f"G/{q}/{sigma}_{op}_{dense}.npy", "wb") as f:
        np.save(f, int_l0)
        np.save(f, int_l1)
        for L_chunk in L_chunks:
            fLS_ = f_L_given_S(L_chunk, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
            np.save(f, fLS_)
            i_ += 1
def MI_int_P(idx, sigma, ce_acc, s_range, prior_s, shares, masked_S, n_shares, q):
    maxhw = int(np.log2(q))+1
    lmin = 0 - sigma*5
    lmax = maxhw + sigma*5
    n_points = int(lmax-lmin)*20
    int_l0, int_l1, int_L = int_l(lmin, lmax, n_points=n_points)
    L0 = int_L[0].reshape(n_points*n_points)
    L1 = int_L[1].reshape(n_points*n_points)
    L = np.vstack((L1, L0))
    fLS_ = f_L_given_S(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
    fLS = fLS_.astype(np.longdouble)
    n_traces = L.shape[1]
    G = np.zeros(n_traces)
    fLS_sum = fLS.sum(axis=0, keepdims=True).squeeze()
    print(f"sigma: {sigma} lmin: {lmin}, lmax: {lmax}, n_points: {n_points}")
    for i, s in enumerate(s_range):
        fLs = fLS[i].squeeze()
        psL = np.where(fLS_sum == 0, 0, fLs/fLS_sum)
        psi = prior_s[i].squeeze()
        g = np.where(psL == 0.0, -0.0, fLs*np.log2(psL))
        G += g
    G_ = G.reshape(n_points, n_points)
    CE = simpson(simpson(G_, int_l1), int_l0)/q
    ce_acc[idx] = ent_s(prior_s) + CE
def MI_int_prec(sigma, s_range, prior_s, shares, masked_S, n_shares, q, op, dense):
    #================Normal setup================
    # pbar = tqdm(range(100))
    #==============Hight noise setup===================
    # pbar = tqdm(range(500))
    with open(f"G/{q}/{sigma}_{op}_{dense}.npy", "rb") as f:
        int_l0 = np.load(f)
        int_l1 = np.load(f)
        n_points = int_l0.shape[0]
        G = np.zeros(n_points**2)
        for i in pbar:
            pbar.set_description(f"Processing chunk {i}")
            fLS_chunky = np.load(f)
            chunk_size = fLS_chunky.shape[1]
            fLS_sum = fLS_chunky.sum(axis=0, keepdims=True).squeeze()
            for i_s, s in enumerate(s_range):
                fLs = fLS_chunky[i_s].squeeze()
                psL = np.where(fLS_sum == 0, 0, fLs/fLS_sum)
                psi = prior_s[i_s].squeeze()
                g = np.where(psL == 0.0, -0.0, fLs*np.log2(psL))
                G[i*chunk_size: (i+1)*chunk_size] += g
    G_ = G.reshape(n_points, n_points)
    CE = simpson(simpson(G_, int_l1), int_l0)/q
    return ent_s(prior_s) + CE



def test_f():
    q = 23
    n_shares = 2
    sigma = 0.03162277660168379
    s_range = np.array([-2, -1, 0, 1, 2])
    n_samples = 4
    ps = prior_ps()
    prior_s = ps[..., np.newaxis, np.newaxis]
    s = np.array([s_range[2]])
    shares = gen_shares_am(s, q, n_shares, n_samples)
    leakages = gen_leakages(shares, n_samples, sigma)
    L = np.array([val for val in leakages.values()])
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    f_S = f_L_given_S(L, s_range, prior_s, shares_sub, masked_S_sub, n_shares, q, sigma)
    print("f_S", f_S.shape)


def test_int():
    q = 23
    n_shares = 2
    sigma = 0.03162277660168379
    s_range = np.array([-2, -1, 0, 1, 2])
    # n_points = 100
    Zq = np.arange(q)
    ps = prior_ps()
    maxhw = HW(Zq).max()
    lmin = 0 - sigma*5
    lmax = maxhw + sigma*5
    n_points = int(lmax-lmin)*50
    # n_points = 100
    print(maxhw, n_points)
    lmin = -1
    lmax = 5
    prior_s = ps[..., np.newaxis, np.newaxis]
    int_l0, int_l1, int_L = int_l(lmin, lmax, n_points=n_points)
    L0 = int_L[0].reshape(n_points*n_points)
    L1 = int_L[1].reshape(n_points*n_points)
    L = np.vstack((L1, L0))
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    G = int_g(L, s_range, prior_s, shares_sub, masked_S_sub, n_shares, q, sigma)
    print(MI_int(q, ps, G, int_l0, int_l1))
def run_MI(q):
    n_shares = 2
    # sigma = 0.03162277660168379
    s_range = np.array([-2, -1, 0, 1, 2])
    maxhw = HW([q])[0]
    ps = prior_ps()

    prior_s = ps[..., np.newaxis, np.newaxis]

    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    sparse_log_mie = np.zeros_like(sparse_sigma_2)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    dense_log_mie = np.linspace(-0.1, -2.75, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    log_mie = np.hstack((sparse_log_mie, dense_log_mie))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    I_sub = np.zeros_like(sigma_2)
    I_add = np.zeros_like(sigma_2)
    i = 0
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    shares_add, masked_S_add = load_precomputed_vals(q, n_shares, op="add", mode="1")

    for s, m in zip(sigma, log_mie):
        lmin = 0 - s*5
        lmax = maxhw + s*5
        n_points = int(lmax-lmin)*50
        int_l0, int_l1, int_L = int_l(lmin, lmax, n_points=n_points)
        L0 = int_L[0].reshape(n_points*n_points)
        L1 = int_L[1].reshape(n_points*n_points)
        L = np.vstack((L1, L0))
        print(f"sigma: {s} lmin: {lmin}, lmax: {lmax}, n_points: {n_points}")
        print("==================OP SUB========================")
        G_sub = int_g(L, s_range, prior_s, shares_sub, masked_S_sub, n_shares, q, s)
        mi_sub = MI_int(q, ps, G_sub, int_l0, int_l1)
        print("==================OP ADD========================")
        G_add = int_g(L, s_range, prior_s, shares_add, masked_S_add, n_shares, q, s)
        mi_add = MI_int(q, ps, G_add, int_l0, int_l1)
        with open(f"log/log_int_{q}_{n_shares}.txt", "a") as f_:
            f_.write(f"{mi_sub}_{mi_add}_{s}\n")
        mi_sub = np.log10(mi_sub)
        mi_add = np.log10(mi_add)
        I_sub[i] = mi_sub
        I_add[i] = mi_add
        i += 1

    with open(f"log/MI_int_{q}_{n_shares}_add.npy", "wb") as f:
        np.save(f, I_add)
    with open(f"log/MI_int_{q}_{n_shares}_sub.npy", "wb") as f:
        np.save(f, I_sub)
    plt.scatter(sigma_2, I_add, color="blue", marker="o", s=10)
    plt.plot(sigma_2, I_add, color="blue", label="s1 = (s + s0)%q")
    plt.scatter(sigma_2, I_sub, color="red", marker="x", s=10)
    plt.plot(sigma_2, I_sub, color="red", label="s1 = (s - s0)%q")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"q={q}")
    plt.savefig(f"pic/MI_int_{q}_{n_shares}.png")
def run_precompute_p(q):
    if os.path.exists(f"G/{q}"):
        print(f"Directory for {q} existed")
    else:
        os.makedirs(f"G/{q}", exist_ok = True)
        print(f"Directory for {q} created")
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    ps = prior_ps()
    prior_s = ps[..., np.newaxis, np.newaxis]
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    ops = tqdm(["sub", "add"])
    for op in ops:
        ops.set_description(f"Processing op {op}")
        shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
        prec = partial(precompute_p, s_range=s_range, prior_s=prior_s, shares=shares, masked_S=masked_S, n_shares=n_shares, q=q, op=op)
        pool = Pool(10)
        pool.map(prec, sigma[13:])
        pool.close()
        pool.join()
    return 0
def run_precompute_p_high(q):
    if os.path.exists(f"G/{q}"):
        print(f"Directory for {q} existed")
    else:
        os.makedirs(f"G/{q}", exist_ok = True)
        print(f"Directory for {q} created")
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    ps = prior_ps()
    prior_s = ps[..., np.newaxis, np.newaxis]
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2_log10 = np.linspace(1.15, 2, 6)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2, sigma_2_log10))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    # print(len(sigma))
    # print(sigma)
    ops = tqdm(["sub", "add"])
    for op in ops:
        ops.set_description(f"Processing op {op}")
        shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
        prec = partial(precompute_p_, s_range=s_range, prior_s=prior_s, shares=shares, masked_S=masked_S, n_shares=n_shares, q=q, op=op)
        pool = Pool(15)
        pool.map(prec, sigma[:15])
        pool.close()
        pool.join()
    for op in ops:
        ops.set_description(f"Processing op {op}")
        shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
        prec = partial(precompute_p_, s_range=s_range, prior_s=prior_s, shares=shares, masked_S=masked_S, n_shares=n_shares, q=q, op=op)
        pool = Pool(7)
        pool.map(prec, sigma[15:22])
        pool.close()
        pool.join()
    for op in ops:
        ops.set_description(f"Processing op {op}")
        shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
        prec = partial(precompute_p_, s_range=s_range, prior_s=prior_s, shares=shares, masked_S=masked_S, n_shares=n_shares, q=q, op=op)
        pool = Pool(4)
        pool.map(prec, sigma[22:26])
        pool.close()
        pool.join()
    for op in ops:
        ops.set_description(f"Processing op {op}")
        shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
        prec = partial(precompute_p_, s_range=s_range, prior_s=prior_s, shares=shares, masked_S=masked_S, n_shares=n_shares, q=q, op=op)
        pool = Pool(3)
        pool.map(prec, sigma[26:])
        pool.close()
        pool.join()
    return 0
def run_MI_p(q):
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    ps = prior_ps()
    prior_s = ps[..., np.newaxis, np.newaxis]
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    I_sub = np.zeros_like(sigma_2)
    I_add = np.zeros_like(sigma_2)
    i = 0
    manager = Manager()
    sigma_low = sigma[:15]
    sigma_high = sigma[15:]
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    shares_add, masked_S_add = load_precomputed_vals(q, n_shares, op="add", mode="1")
    print("==================OP SUB========================")
    CE_low_sub = manager.Array('d', [0.0 for i in range(sigma_low.shape[0])])
    CE_high_sub = manager.Array('d', [0.0 for i in range(sigma_high.shape[0])])
    CE_sub = partial(MI_int_P, ce_acc=CE_low_sub, s_range=s_range, prior_s=prior_s, shares=shares_sub, masked_S=masked_S_sub, n_shares=n_shares, q=q)
    pool = Pool(sigma_low.shape[0])
    pool.starmap(CE_sub, enumerate(sigma_low))
    pool.close()
    pool.join()
    ce_low = np.array(CE_low_sub)
    CE_sub = partial(MI_int_P, ce_acc=CE_high_sub, s_range=s_range, prior_s=prior_s, shares=shares_sub, masked_S=masked_S_sub, n_shares=n_shares, q=q)
    pool = Pool(sigma_high.shape[0])
    pool.starmap(CE_sub, enumerate(sigma_high))
    pool.close()
    pool.join()
    ce_high = np.array(CE_high_sub)
    mi_sub = np.hstack((ce_low, ce_high))
    print("==================OP ADD========================")
    CE_low = manager.Array('d', [0.0 for i in range(sigma_low.shape[0])])
    CE_high = manager.Array('d', [0.0 for i in range(sigma_high.shape[0])])
    CE_add = partial(MI_int_P, ce_acc=CE_low, s_range=s_range, prior_s=prior_s, shares=shares_add, masked_S=masked_S_add, n_shares=n_shares, q=q)
    pool = Pool(sigma_low.shape[0])
    pool.starmap(CE_add, enumerate(sigma_low))
    pool.close()
    pool.join()
    ce_low = np.array(CE_low)
    CE_add = partial(MI_int_P, ce_acc=CE_high, s_range=s_range, prior_s=prior_s, shares=shares_add, masked_S=masked_S_add, n_shares=n_shares, q=q)
    pool = Pool(sigma_high.shape[0])
    pool.starmap(CE_add, enumerate(sigma_high))
    pool.close()
    pool.join()
    ce_high = np.array(CE_high)
    mi_add = np.hstack((ce_low, ce_high))

    with open(f"log/log_int_{q}_{n_shares}.txt", "a") as f_:
        for i in range(23):
            f_.write(f"{mi_sub[i]}_{mi_add[i]}_{sigma[i]}\n")

    mi_sub = np.log10(mi_sub)
    mi_add = np.log10(mi_add)

    with open(f"log/MI_int_{q}_{n_shares}_add.npy", "wb") as f:
        np.save(f, mi_add)
    with open(f"log/MI_int_{q}_{n_shares}_sub.npy", "wb") as f:
        np.save(f, mi_sub)
    plt.figure(figsize=(12,9), dpi=180)
    plt.scatter(sigma_2, mi_add, color="blue", marker="o", s=10)
    plt.plot(sigma_2, mi_add, color="blue", label="s1 = (s + s0)%q")
    plt.scatter(sigma_2, mi_sub, color="red", marker="x", s=10)
    plt.plot(sigma_2, mi_sub, color="red", label="s1 = (s - s0)%q")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"q={q}")
    plt.savefig(f"pic/MI_int_{q}_{n_shares}.png")

    return 0

def run_MI_prec(q):
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    ps = prior_ps()
    prior_s = ps[..., np.newaxis, np.newaxis]
    #===============Normal setup===================
    # sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    # dense_sigma_2 = np.linspace(-1.25, 1, 19)
    # sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    # sigma_2_10 = np.power(10, sigma_2)
    # sigma = np.sqrt(sigma_2_10)

    #===============Highnoise setup===================
    # sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    # dense_sigma_2 = np.linspace(-1.25, 1, 19)
    # sigma_2_log10 = np.linspace(1.15, 2, 6)
    # sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2, sigma_2_log10))
    # sigma_2_10 = np.power(10, sigma_2)
    # sigma = np.sqrt(sigma_2_10)


    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    shares_add, masked_S_add = load_precomputed_vals(q, n_shares, op="add", mode="1")
    mi_sub = np.empty_like(sigma, dtype=np.longdouble)
    mi_add = np.empty_like(sigma, dtype=np.longdouble)
    pbar = tqdm(enumerate(sigma))
    for i, sig in enumerate(sigma):
        pbar.set_description(f"Process for sigma: {sig:0.4f}")
        pbar.set_postfix_str(s='sub', refresh=True)
        op = "sub"
        mi_sub[i] = MI_int_prec(sig, s_range, prior_s, shares_sub, masked_S_sub, n_shares, q, op, dense=50)
        pbar.set_postfix_str(s='add', refresh=True)
        op = "add"
        mi_add[i] = MI_int_prec(sig, s_range, prior_s, shares_add, masked_S_add, n_shares, q, op, dense=50)
        with open(f"log/log_int_{q}_{n_shares}.txt", "a") as f_:
                f_.write(f"{mi_sub[i]}_{mi_add[i]}_{sig}\n")
    mi_sub = np.log10(mi_sub)
    mi_add = np.log10(mi_add)

    with open(f"log/MI_int_{q}_{n_shares}_add.npy", "wb") as f:
        np.save(f, mi_add)
    with open(f"log/MI_int_{q}_{n_shares}_sub.npy", "wb") as f:
        np.save(f, mi_sub)
    plt.figure(figsize=(12,9), dpi=180)
    plt.scatter(sigma_2, mi_add, color="blue", marker="o", s=10)
    plt.plot(sigma_2, mi_add, color="blue", label="s1 = (s + s0)%q")
    plt.scatter(sigma_2, mi_sub, color="red", marker="x", s=10)
    plt.plot(sigma_2, mi_sub, color="red", label="s1 = (s - s0)%q")
    plt.annotate(round(mi_add[0], 4),(-3,mi_add[0]),textcoords="offset points", color="blue",xytext=(0,10),ha='center')
    plt.annotate(round(mi_sub[0], 4),(-3,mi_sub[0]),textcoords="offset points", color="red",xytext=(0,10),ha='center')
    plt.annotate(round(mi_add[-1], 4),(2,mi_add[-1]),textcoords="offset points", color="blue",xytext=(12,10),ha='center')
    plt.annotate(round(mi_sub[-1], 4),(2,mi_sub[-1]),textcoords="offset points", color="red",xytext=(12,10),ha='center')
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.yticks(np.linspace(-0.5, -5.5, 11))
    plt.legend()
    plt.title(f"q={q}")
    plt.savefig(f"pic/MI_int_{q}_{n_shares}.png")

def convergence(q):
    n_shares=2
    plt.figure(figsize=(12,3), dpi=180)
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    n_seeds = [100, 1000, 10000, 50000, 100000, 500000, 1000000]
    C = [ '#00ffff', '#0080ff', '#ffbf00', '#40ff00', '#0099ff', '#8000ff', '#ff00bf']
    for i, n_s in enumerate(n_seeds):
        with open(f"log/conv_{q}_{sigma[0]}_{n_s}.npy", "rb") as f:
            mi_n = np.load(f)
        mean = mi_n.mean()
        stdm = np.std(mi_n)
        print(f"n_samples: {n_s}, mean: {mean}, var: {stdm}")
        plt.scatter(mean, 0, marker='x', s=15, color=C[i], label=f"{n_s} {mean:0.5f} {stdm:0.5f}")
        plt.errorbar(mean, 0, xerr=stdm, ecolor=C[i], capsize=2, alpha=0.7)
    plt.scatter(0.49868769436767213, 0, color='red', label=f"integral 0.49868", marker="x")
        # plt.plot(sigma_2, mi_n, color=C[i], label=f"n_samples: {n_s}")
    # with open(f"log/MI_int_{q}_{n_shares}_{op}.npy", "rb") as f:
    #     mi_int = np.load(f)
    # plt.plot(sigma_2, mi_int, color="red", label=f"integral")
    # plt.xlabel("$\log_{10}(\sigma^{2})$")
    # plt.ylabel("$\log_{10}(MI)$")
    plt.ylim(-0.01, 0.01)
    plt.legend(fontsize=4)
    plt.title(f"q={q}")
    plt.show()
    # plt.savefig(f"pic/MI_convergence_{q}_{n_shares}.png")
def check_add():
    plt.figure(figsize=(12,9), dpi=180)
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    qs = [5, 13, 17, 19, 23, 1009, 1021, 3329]
    op = 0
    ops = ["sub", "add"]
    ops_des = ["s1 = (s-s0)%q", "s1 = (s+s0')%q"]
    for Q in qs:
        with open(f"log/MI_int_{Q}_2_{ops[op]}.npy", "rb") as f:
            mi = np.load(f)[:23]
        plt.plot(sigma_2, mi, label=f"q={Q}_{bin(Q)[2:]}")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"{ops_des[op]}")
    plt.savefig(f"pic/MI_{ops[op]}_int_{qs}.png")
if __name__ == '__main__':
    # test_f()
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # test_int()
    # st = time()

    # print("=============Normal run=============\n", time()-st)
    # st = time()
    # q=3329
    # Zq = np.arange(q)
    # hw = HW(Zq)
    # print(hw.max())


    #==============On the fly parallel MI compute===============
    # run_MI_p(q=23)
    #==============+RUN MI w/ precomputed values=============
    # run_precompute_p_high(q=3329)
    # run_MI_prec(q=3329)

    #==============Convergence check=================
    # convergence(q=23)

    #================Check consistency===============
    check_add()
    # print("===============P run===========\n", time()-st)
    # prior_s = prior_ps()
    # prior_s_ = prior_s[..., np.newaxis, np.newaxis]
    # print("prior_s", prior_s.shape, prior_s_.shape)
    # s_0 = np.array([[ 4.69030088],  [26.60698543],  [25.57159243],  [ 6.44741229]])
    # print(s_0.shape, s_0*0.0625)
