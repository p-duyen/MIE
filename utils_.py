import numpy as np
import itertools as it
from scipy.stats import binom


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

def gen_all_shares_S(s_range, q, n_shares, op="sub", mode=1):
    """Exhaustive shares matrix for every y in Y

    Parameters
    ----------
    s_range : array
        Possible values for s
    q: int
        Prime field
    n_shares: int
        Number of shares in encoding
    op: string "add" or "sub"
        Operations to compute masked s (s + (q - sum_mod(shares)) for "add", (s - sum_mod(shares) for "sub"))
    mode: int 0, 1
        For add op: 0 to convert every shares to the additive inverse, 1 to convert 1 share to additive inverse

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

def load_precomputed_vals(q, n_shares, op, mode):
    with open(f"precompute_vals/precomputed_shares_am_{q}_{n_shares}shares_{op}_mode{mode}.npy", "rb") as f:
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

def pretty_print_a(a, tag=None):
    if tag != None:
        print(tag)
    for i in a:
        print(i, end=", ")

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
