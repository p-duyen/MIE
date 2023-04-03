import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *
from time import time
from multiprocessing import Pool, Manager, Value, Array, Lock
from functools import partial
from scipy.integrate import simpson
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock
np.seterr(invalid="ignore")

def tile_(li, iof, lock,  L0, L1):
    """Tile the values of the 3rd shares to the precomputed first 2 shares and write it to file,
    concurrently and orderly, using thread lock.
    Avoid mem overflow when trying to gen 3 shares all at once
    Parameters
    ----------
    li: float
        One value in predefined leakage range to tile
    iof: file
        File to write the results
    lock: thread lock
        Lock to synch the writing of results (avoid unorder map)
    L0, L1: array shape (n_points^2)
         Predefined 2 first shares
    Returns
    -------
    None
    The results are written into file corresp. to sigma and q and density.

    """
    L2 = np.ones_like(L0)*li
    L =  np.vstack((L0, L1, L2))
    with lock:
        np.save(iof, L)

def int_L_3_p(q, sigma, n_chunks=5, n_shares=3, dense=50):
    """ Generate exhaustive values of leakages for 3 shares and save to file corresp. to sigma value.
    L \in [-5sigma, maxHW(Fq) + 5sigma] => |L| = (maxHW(q) + 10*sigma )*dense = n_points
    Same idea w/ 2 shares case but to avoid mem overflowed:
    - Gen exhaustive leakages for 2 shares
    - Tile the 3rd share in the end w/ corresp. first 2 shares.
    - Write tiled complete 3 shares tupple into file using parallel threads.
    Parameters
    ----------
    q: int
        Prime. Use to compute upper limit for leakage range
    sigma: float
        Noise level, needed to decide the range of exhaustive search traces
    n_chunks: int
        Number of concurrent threads to write results to file
    dense: int
         Density of leakage range.
    Returns
    -------
    None
    The results are written into file corresp. to sigma and q and density.
    """
    Zq = np.arange(q)
    hw = HW(Zq)
    maxhw = hw.max()
    lmin = - sigma*5
    lmax = maxhw + sigma*5
    n_points = int(lmax-lmin)*dense
    dir_path = f"precompute_vals/int_L/{q}"
    if os.path.exists(dir_path):
        print(f"Directory for {q} existed")
    else:
        os.makedirs(dir_path, exist_ok = True)
        print(f"Directory for {q} created")
    fn = f"{n_shares}_{dense}_{sigma}_p.npy"
    if os.path.exists(f"{dir_path}/{fn}"):
        print(f"Data's ready for {q} {sigma:0.4f}")
        return 0
    int_li, int_L_2 = int_l(lmin, lmax, n_points)
    L0 = int_L_2[0].reshape(n_points*n_points)
    L1 = int_L_2[1].reshape(n_points*n_points)
    int_li_chunks = np.array_split(int_li, n_points/n_chunks)
    int_pbar = tqdm(int_li_chunks)

    with open(f"{dir_path}/{fn}", "wb") as f:
        np.save(f, int_li)
        for li_chunk in int_pbar:
            lock = Lock()
            int_pbar.set_description(f"Precompute L for {q} sigma: {sigma:0.4f}")
            f_tile = partial(tile_, iof=f, lock=lock, L0=L0, L1=L1)
            with ThreadPoolExecutor(n_chunks) as executer:
                executer.map(f_tile, li_chunk)
                # np.save(np.array(list(results)))
    print(f"Data's ready for {q} {sigma:0.4f}")

def f_L_given_S_HW(L, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
    """Compute f(l|s) for all possible value of the secret s and multiple traces
    f(l|s) = sum_s1 p(s1)f(l0|s0)f(l1|s1)
    Value of shares are HW of them instead of values in F_q

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
    n_share_vals =  shares.shape[0]
    Zq = np.arange(q)
    hw_set = HW(Zq)
    hw_set_u = np.unique(hw_set)


    f_L_given_si = np.ones((n_traces, n_share_vals))

    for ns in range(n_shares-1):
        Li = np.expand_dims(L[ns], axis=1)
        f_Li = np.apply_along_axis(pdf_normal, axis=1, arr=Li, mu=hw_set_u, sigma=sigma)
        f_Li = f_Li[:, shares[:, ns]]
        f_L_given_si  *= f_Li
    L0 = np.expand_dims(L[n_shares-1], axis=1)
    f_L0 = np.apply_along_axis(pdf_normal, axis=1, arr=L0, mu=hw_set_u, sigma=sigma)
    fLS = []
    for ms in masked_S:
        f_L0_ms = f_L0[:, ms]
        f_L_given_S_prod = f_L_given_si*f_L0_ms
        f_L_given_S_prod_sum = f_L_given_S_prod.sum(axis=1)
        fLS.append(f_L_given_S_prod_sum)
    fLS_ = np.array(fLS)
    fLS_ = prior_s*fLS_
    return fLS_

def int_g_worker(L, s_range, prior_s, shares_chunks, masked_chunks, n_shares, q, sigma):
    """Worker to put in parallel processing. Precompute values under integral
    g = sum_s p(s) f(l|s) log2 p(s|l) sum over every shares and multiple traces L (i.e different chunks of L at the same time)
    The whole shares values range is split into smaller chunks and get them accumulated
    Extendable

    Parameters
    ----------
    L: array shape (3, n_points)
        a chunk of n_points leakage traces (3 entries corresp. to 3 shares)
    s_range : array size S
        Possible values for s. Set to [-2, -1, 0, 1, 2] in current case but can extend to uniform if needed
    prior_s : array size S
        Prior proba of s (CBD)
    shareschunks: array shape: (n_chunks, n_shares-1, q^(n_shares-1))
        Precomputed and split into smaller chunks of all possible randomness (l1, ..., ld-2)
    masked_chunks: array shape: (n_chunks, S, q^(n_shares-1))
        Precomputed and split into smaller chunks of masked secret (l0) for all possible value of secret
    n_shares: int
        Number of shares. Work w/ 2 and 3, could extend to higher number of shares of the format of precomputed L is compatible
    q: int
        Prime
    sigma: float
        Noise level

    Returns
    -------
    G = sum_s p(s) f(l|s) log2 p(s|l) sum over every shares and L
    Ready to collect and save in master function (precompute_f_p)

    """

    s_size = s_range.shape[0]
    n_traces = L.shape[1]
    fLS_acc = np.zeros((s_size, n_traces))
    for s_chunk, ms_chunk in zip(shares_chunks, masked_chunks):
        fLS_acc += f_L_given_S_HW(L, s_range, prior_s, s_chunk, ms_chunk, n_shares, q, sigma)
    G = np.zeros(n_traces)
    fLS_sum = fLS_acc.sum(axis=0, keepdims=True).squeeze()
    for i, s in enumerate(s_range):
        fLs = fLS_acc[i].squeeze()
        psL = np.where(fLS_sum == 0, 0.0, fLs/fLS_sum)
        log2_psL = np.ones_like(psL)*-0.0
        log2_psL[np.nonzero(psL)] = np.log2(psL[np.nonzero(psL)])
        G += fLs*log2_psL
    return G




def precompute_f_p(sigma, s_range, prior_s, n_shares, q, op, mode, p_chunks=10, s_chunks=4, dense=20):
    """Precompute values under integral
    g = sum_s p(s) f(l|s) log2 p(s|l) sum over every shares and leakage range.
    To avoid mem overflow (when extending to (n_traces, shares_val) i.e (n_points^3, q^3)):
    - Precompted all shares values and all leakage value ranges.
    - Load all shares and masked secret at once (cost q^2)
    - Load leakage tupple chunk by chunk
    - Chop the big chunk of leakage range into smaller chunks and process them in parallel
    - Chop the whole shares values range into smaller chunks and get them accumulated (couldn't parallel this part)
    g value for each small chunk of leakages will be written in order for later integral.
    Parameters
    ----------
    sigma: float
        Standard deviation for Gaussian leakages
    s_range : array size S
        Possible values for s. Set to [-2, -1, 0, 1, 2] in current case but can extend to uniform if needed
    prior_s : array size S
        Prior proba of s (CBD)
    n_shares: int
        Number of shares. Work w/ 2 and 3, could extend to higher number of shares of the format of precomputed L is compatible
    q: int
        Prime
    op: string "add" or "sub"

    mode: string "one" or "all"
    p_chunks: int
        Number of concurrent processes used to compute g each process processes data size max (p_chunks*n_points, q^2/s_chunk)
        Chose wisely because it doesn't scale up linearly w/ speed (bottleneck w/ file write operations)
        And also avoid mem overflow
    s_chunks: int
        Number of chunks that the whole shares values got chopped up.
        Larger means lighter but longer process.
    dense: int
        Density of leakage range.
        Just a flag to get the right file. No effect to computation (already precompute)


    Returns
    -------
    None
    g value for each chunk of leakage is written into file in right order for integral
    """
    shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode=mode)
    prior_s = prior_s[..., np.newaxis]
    dir_path = f"precompute_vals/G/{q}"
    if os.path.exists(dir_path):
        print(f"Directory for {q} existed")
    else:
        os.makedirs(dir_path, exist_ok = True)
        print(f"Directory for {q} created")
    mode_flag = "" if op == "sub" else mode

    shares_chunks = np.array_split(shares, s_chunks, axis=0)
    masked_chunks = np.array_split(masked_S, s_chunks, axis=1)
    s_size = s_range.shape[0]
    with open(f"precompute_vals/int_L/{q}/{n_shares}_{dense}_{sigma}_p.npy", "rb") as f:
        int_li = np.load(f)
        fn_g = f"{dir_path}/{n_shares} shares_{sigma}_{op}{mode_flag}_p.npy"
        with open(fn_g, "wb") as f_g:
            np.save(f_g, int_li)
            n_chunks = int_li.shape[0]
            outer_pbar = tqdm(range(n_chunks), desc=f"Processing {op} {mode} {sigma:0.4f}", leave=True)
            for chunk_i in outer_pbar:
                L = np.load(f)
                L_chunks = np.split(L, n_chunks, axis=1)

                int_g_p = partial(int_g_worker, s_range=s_range, prior_s=prior_s, shares_chunks=shares_chunks, masked_chunks=masked_chunks, n_shares=n_shares, q=q, sigma=sigma)
                with Pool(p_chunks) as pool:
                    res = pool.map_async(int_g_p, L_chunks, chunksize=p_chunks)
                    for val in res.get():
                        np.save(f_g, val)
def MI_int_prec_3_p(sigma, s_range, prior_s, n_shares, q, op, mode, dense=20):
    """Compute MI with precomputed values of g.
    Collect value of G into corresp. shape of numerical integral (n_points, n_points, n_points) for 3 shares.
    Compute MI = H(S) - int_L (G) using Simpson's rule
    Uniquely for 3 shares.
    Return MI value
    """
    mode_flag = "" if op == "sub" else mode
    print(f"Compute MI for sigma {sigma} op {op} mode {mode}", end=" ")
    with open(f"precompute_vals/G/{q}/{n_shares} shares_{sigma}_{op}{mode_flag}_p.npy", "rb") as f_g:
        int_li = np.load(f_g)
        n_points = int_li.shape[0]
        G = np.empty((n_points, n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                G[i, j] = np.load(f_g)
    CE = simpson(simpson(simpson(G, int_li), int_li), int_li)/(q**(n_shares-1))
    print("Done")
    return ent_s(prior_s) + CE

def prec_run(q, sigma):
    """
    Precompute g value for all ops and modes and sigma.

    """
    n_shares = 3
    s_range = np.array([-2, -1, 0, 1, 2])
    prior_s = prior_ps()
    ops = ["sub", "add"]
    modes = ["one", "all"]
    n_chunks = 10 # Number of concurrent threads for writing int_L to file
                # each thread process data size (n_points^2, n_points)
    p_chunks = 15 # Number of concurrent processes used to compute int_g
                    # each process processes data size max (p_chunk_size*n_points, q^2/s_chunk)
    s_chunks = 1 # Split all shares (q^2) to s_chunk subarray, each subarray has size max q^2/s_chunk
    for sig in sigma:
        int_L_3_p(q, sig, n_chunks=n_chunks, n_shares=3, dense=20)
    for o in ops:
        for m in modes:
            if (o == "sub" and m =="one") :
                pass
            else:
                gen_all_shares_S_HW(s_range, q, n_shares, o, m)
                for sig in sigma:
                    precompute_f_p(sigma=sig, s_range=s_range, prior_s=prior_s, n_shares=n_shares, q=q, op=o, mode=m, p_chunks=p_chunks, s_chunks=s_chunks, dense=20)
def MI_run(q, sigma_2, sigma):
    """Compute MI from precomputed G for all ops and modes and sigmas.
    Save value for corresp. op and mode with all sigma in predefined range.
    Plot MI curves for different ops and modes.
    """
    n_shares = 3
    s_range = np.array([-2, -1, 0, 1, 2])
    prior_s = prior_ps()
    ops = ["sub", "add"]
    modes = ["one", "all"]
    for o in ops:
        for m in modes:
            if (o == "sub" and m =="one") :
                pass
            else:
                mi = np.empty_like(sigma)
                for i, sig in enumerate(sigma):
                    mi[i] = np.log10(MI_int_prec_3_p(sig, s_range, prior_s, n_shares, q, op=o, mode=m, dense=20))
                with open(f"log/MI_int_{q}_{n_shares}_{o}_{m}.npy", "wb") as f:
                    np.save(f, mi)
                plt.plot(sigma_2, mi, label=f"{n_shares} shares {o} {m}")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    q = 23
    sigma_2, sigma = gen_sigma()
    prec_run(q, sigma[:-1])
    MI_run(q, sigma_2[:-2], sigma[:-2])
