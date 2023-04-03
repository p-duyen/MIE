import numpy as np
import itertools as it
from scipy.stats import binom
import os
from tqdm import tqdm


def count_1(x):
    return int(x).bit_count()

def rep_bin(x):
    return f"{x:08b}"

fbin = np.vectorize(rep_bin)

def BIN(x):
    return fbin(x)

fcount = np.vectorize(count_1)
def HW(x):
    return fcount(x)

def shares_addmod(x, q):
    sum_x = x.sum()
    return sum_x%q
# faddmod = np.vectorize(shares_addmod)

def pdf_normal(x, mu, sigma):
    ep = (x-mu)/sigma
    ep = -ep**2/2
    return np.exp(ep) / (sigma * np.sqrt(2*np.pi))


def gen_shares(y, n_shares, n_samples):
    shares = {}
    masked_y = y.copy()
    if n_shares <=1 :
        shares["Y"] = masked_y
        return shares
    else:
        for i in range(1, n_shares):
            share_values = np.random.randint(0, 256, (n_samples, ))
            shares[f"X_{i}"] = share_values
            masked_y = masked_y^share_values
        shares[f"X_{0}"] = masked_y
        return shares


def gen_leakages(shares, n_samples, sigma):
    leakages = {}
    for key, val in shares.items():
        leakages[key] = HW(val) + np.random.normal(0, sigma, (n_samples))
    return leakages


def gen_shares_am(s, q, n_shares, n_samples, seed=12345, op="sub", mode=1):
    """Draw random shares for a secret value s

    Parameters
    ----------
    s: int
        Secret value to gen corresp. shares
    q: int
        Prime
    n_shares: int
        Number of shares in encoding
    n_samples: int
        Number of samples
    seed: int
        Seed for prg, force different op and mode to have the same shares values
    op: string "add" or "sub"
        Operations to compute masked s (s + (q - sum_mod(shares)) for "add", s - sum_mod(shares) for "sub"))
    mode: int 0, 1
        For add op: 0 to convert every shares to the additive inverse, 1 to convert 1 share to additive inverse

    Returns
    -------
        shares: dict
        Corresp. shares for s
        X_i <--$ [0, q] n_samples times
        X_0 = (s - add_mod(X_i, q))%q
        if op == "add" & mode == 1:
            X_1 will be converted to additive inverse
        if op == "add" & mode == 0:
            all X_i will be converted to additive inverse
    """
    rng = np.random.default_rng(seed) # reproduce same shares for different ops and modes
    shares = {}
    sum_shares = np.zeros((n_samples, ))
    for i in range(1, n_shares):
        share_values = rng.integers(low=0, high=q, size=(n_samples, ))
        sum_shares = (sum_shares + share_values)%q
        shares[f"S_{i}"] =  q - share_values if (op=="add" and mode==0) else share_values
    shares[f"S_{0}"] = (s - sum_shares)%q
    if op=="add" and mode==1:
            shares["S_1"] = (q - shares["S_1"])%q
    return shares

def gen_all_shares_S_(s_range, q, n_shares, op="sub", mode=1):
    """Exhaustive shares matrix for every s in s_range
        s = (s0 + s1 + ... + s{d - 1})%q
        s1, ..., s_{d-1} <--$ F_q
        s0 = [s - (s1,..., s_{d-1})] %q
    Parameters
    ----------
    s_range : array
        Possible values for s
    q: int
        Prime field
    n_shares: int
        Number of shares in encoding
    op: string "add" or "sub"
        Operations choose to represent shares ()
        op == "sub":
            Keep s_1, ..., s_{d-1} as generated and subtract the sum of shares.
        op == "add"
            Change s1, ..., s_{d-1} to (q-si) depends on the mode and sum (sub some of them) the shares.
    mode: int 0, 1
        Option for number of shares that changes it representation.
        mode == 1:
            Only s1 change into q-s1
        mode == 0:
            All shares will change into q-s_i
    Returns
    -------
    all_shares: array shape: (n_shares-1, q^(n_shares-1))
        All posible n_shares-1 shares values
    masked_S: array shape: (s_range, q^(n_shares-1))
        Masked secret for all value in s_range,
    """
    size_S = s_range.shape[0]
    Zq = np.arange(q).reshape(1, q)
    share_range = np.repeat(Zq, repeats=n_shares-1, axis=0)
    all_shares = np.zeros((q**(n_shares-1), n_shares-1), dtype=np.int16)
    masked_S = np.zeros((size_S, q**(n_shares-1)), dtype=np.int16)
    share_vals = np.meshgrid(*share_range)
    for i in range(n_shares-1):
        all_shares[..., i] = share_vals[i].reshape(q**(n_shares-1))
    for i, s in enumerate(s_range):
        shares_sum = np.apply_along_axis(shares_addmod, 1, all_shares, q)
        masked_S[i] = (s - shares_sum)%q
    if op=="add":
        if mode==1:
            all_shares[..., 0] = (q - all_shares[..., 0])%q
        if mode==0:
            all_shares = (q - all_shares)%q
    with open(f"precompute_vals/precomputed_shares_am_{q}_{n_shares}shares_{op}_mode{mode}.npy", "wb") as f:
        np.save(f, all_shares)
        np.save(f, masked_S)
    return all_shares, masked_S
def gen_all_shares_S(s_range, q, n_shares, op="sub", mode=1):
    """Exhaustive shares matrix for every s in s_range
        s = (s0 + s1 + ... + s{d - 1})%q
        s1, ..., s_{d-1} <--$ F_q
        s0 = [s - (s1,..., s_{d-1})] %q
    Parameters
    ----------
    s_range : array
        Possible values for s
    q: int
        Prime field
    n_shares: int
        Number of shares in encoding
    op: string "add" or "sub"
        Operations choose to represent shares
        op == "sub":
            Keep s_1, ..., s_{d-1} as generated and subtract the sum of shares.
        op == "add"
            Change s1, ..., s_{d-1} to (q-si) depends on the mode and sum (sub some of them) the shares.
    mode: string "one" or "all"
        Option for number of shares that changes it representation.
        mode == "one":
            Only s1 change into q-s1
        mode == "all":
            All shares will change into q-s_i
    Returns
    -------
    all_shares: array shape: (n_shares-1, q^(n_shares-1))
        All posible n_shares-1 shares values
    masked_S: array shape: (s_range, q^(n_shares-1))
        Masked secret for all value in s_range,
    """
    size_S = s_range.shape[0]
    Zq = np.arange(q).reshape(1, q)
    share_range = np.repeat(Zq, repeats=n_shares-1, axis=0)
    all_shares = np.zeros((q**(n_shares-1), n_shares-1), dtype=np.int16)
    masked_S = np.zeros((size_S, q**(n_shares-1)), dtype=np.int16)
    share_vals = np.meshgrid(*share_range)
    for i in range(n_shares-1):
        all_shares[..., i] = share_vals[i].reshape(q**(n_shares-1))
    for i, s in enumerate(s_range):
        shares_sum = np.apply_along_axis(shares_addmod, 1, all_shares, q)
        masked_S[i] = (s - shares_sum)%q
    if op=="add":
        if mode=="one":
            all_shares[..., 0] = (q - all_shares[..., 0])%q
        if mode=="all":
            all_shares = (q - all_shares)%q
    mode_flag = "" if op == "sub" else mode
    with open(f"precompute_vals/precomputed_shares_am_{q}_{n_shares}shares_{op}{mode_flag}.npy", "wb") as f:
        np.save(f, all_shares)
        np.save(f, masked_S)
    print(f"=============Gen shares for {q} {n_shares} shares op {op} mode {mode} DONE===============")
    return all_shares, masked_S
def gen_all_shares_S_HW(s_range, q, n_shares, op="sub", mode=1):
    """Exhaustive shares matrix for every s in s_range
        s = (s0 + s1 + ... + s{d - 1})%q
        s1, ..., s_{d-1} <--$ F_q
        s0 = [s - (s1,..., s_{d-1})] %q
    Parameters
    ----------
    s_range : array
        Possible values for s
    q: int
        Prime field
    n_shares: int
        Number of shares in encoding
    op: string "add" or "sub"
        Operations choose to represent shares
        op == "sub":
            Keep s_1, ..., s_{d-1} as generated and subtract the sum of shares.
        op == "add"
            Change s1, ..., s_{d-1} to (q-si) depends on the mode and sum (sub some of them) the shares.
    mode: string "one" or "all"
        Option for number of shares that changes it representation.
        mode == "one":
            Only s1 change into q-s1
        mode == "all":
            All shares will change into q-s_i
    Returns
    -------
    all_shares: array shape: (n_shares-1, q^(n_shares-1))
        All posible n_shares-1 shares values
    masked_S: array shape: (s_range, q^(n_shares-1))
        Masked secret for all value in s_range,
    """
    mode_flag = "" if op == "sub" else mode
    fn = f"precompute_vals/shares/{q}_{n_shares}shares_{op}{mode_flag}.npy"
    if os.path.exists(fn):
        print(f"Shares values are ready for {q} {n_shares} shares op {op} mode {mode}")
        return 0
    size_S = s_range.shape[0]
    Zq = np.arange(q).reshape(1, q)
    share_range = np.repeat(Zq, repeats=n_shares-1, axis=0)
    all_shares = np.zeros((q**(n_shares-1), n_shares-1), dtype=np.int16)
    masked_S = np.zeros((size_S, q**(n_shares-1)), dtype=np.int16)
    share_vals = np.meshgrid(*share_range)
    for i in range(n_shares-1):
        all_shares[..., i] = share_vals[i].reshape(q**(n_shares-1))
    for i, s in enumerate(s_range):
        shares_sum = np.apply_along_axis(shares_addmod, 1, all_shares, q)
        masked_S[i] = (s - shares_sum)%q
    if op=="add":
        if mode=="one":
            all_shares[..., 0] = (q - all_shares[..., 0])%q
        if mode=="all":
            all_shares = (q - all_shares)%q
    with open(fn, "wb") as f:
        np.save(f, HW(all_shares))
        np.save(f, HW(masked_S))
    print(f"Shares values are ready for {q} {n_shares} shares op {op} mode {mode}")
def ent_s(prior_s):
    """Entropy for prior proba
    """
    log_2p = np.log2(prior_s)
    return -(prior_s*log_2p).sum()

def load_precomputed_vals(q, n_shares, op, mode):
    mode_flag = "" if op == "sub" else mode
    with open(f"precompute_vals/shares/{q}_{n_shares}shares_{op}{mode_flag}.npy", "rb") as f:
        shares = np.load(f)
        masked_S = np.load(f)
        return shares, masked_S

def pretty_print_dict(d, tag=None):
    if tag:
        print(tag)
    for key, val in d.items():
        print(f"{key}: {val}")
def pretty_print_dict_HW(d, tag=None):
    if tag:
        print(tag)
    for key, val in d.items():
        print(f"{key}: {HW(val)}")

def pretty_print_a(a, tag=None, end=", "):
    if tag != None:
        print(tag)
    for i in a:
        print(i, end=end)

def prior_ps():
    a = binom(2, 0.5)
    b = binom(2, 0.5)
    s_range = np.array([-2, -1, 0, 1, 2])
    ps = np.zeros((5, ))
    for i, s in enumerate(s_range):
        p = 0
        for i_a in range(3):
            for i_b in range(3):
                if i_a - i_b==s:
                    p += a.pmf(i_a)*b.pmf(i_b)
        ps[i] = p
    return ps
def int_l(lmin=-2, lmax=7, n_points=100):
    int_l0 =  np.linspace(lmin, lmax, num=n_points, endpoint=True)
    int_l1 =  np.linspace(lmin, lmax, num=n_points, endpoint=True)
    int_L = np.meshgrid(int_l0, int_l1)

    return int_l0, int_L

def int_L_3(q, sigma, n_shares=3, dense=50):
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
    fn = f"{n_shares}_{dense}_{sigma}.npy"
    if os.path.exists(f"{dir_path}/{fn}"):
        print(f"Data's ready for {q} {sigma:0.4f}")
        return 0
    int_li, int_L_2 = int_l(lmin, lmax, n_points)
    L0 = int_L_2[0].reshape(n_points*n_points)
    L1 = int_L_2[1].reshape(n_points*n_points)
    print("loop size", int_li.shape)
    int_pbar = tqdm(int_li)
    with open(f"{dir_path}/{fn}", "wb") as f:
        np.save(f, int_li)
        for li in int_pbar:
            int_pbar.set_description(f"Precompute L for {q} sigma: {sigma:0.4f}")
            L2 = np.ones_like(L0)*li
            L =  np.vstack((L0, L1, L2))
            np.save(f, L)
    print(f"Data's ready for {q} {sigma:0.4f}")

def gen_sigma_():
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2_log10 = np.linspace(1.25, 2, 4)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2, sigma_2_log10))
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    return sigma_2, sigma
def gen_sigma():
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2_log10 = np.linspace(1.25, 2, 4)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2, sigma_2_log10))
    sigma_2_10 = np.power(10, sigma_2)
    idx = range(0, len(sigma_2), 2)
    sigma = np.sqrt(sigma_2_10)
    return sigma_2[idx], sigma[idx]
# print(gen_sigma())
