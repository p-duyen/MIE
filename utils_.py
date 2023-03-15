import numpy as np
import itertools as it


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


def gen_all_shares(y, n_shares, y_range):
    y_range_ = np.arange(y_range).reshape(1, 256)
    if n_shares <= 1:
        all_shares = y.copy()
    else:
        share_range = np.repeat(y_range_, repeats=n_shares-1, axis=0)
        all_shares = it.product(*share_range)
    return list(all_shares)

def gen_shares_exhaustive(y, n_shares, y_range):
    y_range_ = np.arange(y_range).reshape(1, 256)
    if n_shares <= 1:
        all_shares = y.copy()
    else:
        share_range = np.repeat(y_range_, repeats=n_shares-1, axis=0)
        all_shares = it.product(*share_range)
    return list(all_shares)

def pretty_print_dict(d, tag=None):
    if tag:
        print(tag)
    for key, val in d.items():
        print(f"{key}: {val}")

def pretty_print_a(a, tag=None):
    if tag != None:
        print(tag)
    for i in a:
        print(i, end="\n")
