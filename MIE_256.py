import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *
from time import time

def r_xor(x):
    return np.bitwise_xor.reduce(x)
fxor = np.vectorize(np.bitwise_xor)

def gen_all_shares_Y(y_range, n_shares):
    """
    Exhaustive shares matrix for every y in Y

    return
    all_shares: all posible n_shares-1 shares values, shape: (n_shares-1, y_range^(n_shares-1))
    masked_Y: masked secret for all value in Y, shape: (y_range, y_range^(n_shares-1))
    """
    y_range_ = np.arange(y_range).reshape(1, 256)
    all_shares = np.zeros((y_range**(n_shares-1), n_shares-1), dtype=np.int16)
    share_range = np.repeat(y_range_, repeats=n_shares-1, axis=0)
    masked_Y = np.zeros((y_range, y_range**(n_shares-1)), dtype=np.int16)
    share_vals = np.meshgrid(*share_range)
    for i in range(n_shares-1):
        all_shares[..., i] = share_vals[i].reshape(y_range**(n_shares-1))
    for y in range(y_range):
        masked_Y[y] = np.apply_along_axis(r_xor, 1, all_shares)^y
    return all_shares, masked_Y


def f_li_given_Y(li, y_range, sigma):
    """f(li|y) for all y in Y"""
    y_range_ = np.arange(y_range)
    hw_set = HW(y_range_)
    hw_set_u = np.unique(hw_set)
    li_ = np.repeat(li, repeats=hw_set_u.shape[0], axis=0)
    pdf_hw = pdf_normal(li_, hw_set_u, sigma)
    f_li = pdf_hw[hw_set]
    return f_li


def f_l_given_Y(l, shares, masked_Y, n_shares, y_range, sigma):
    """
    f(l|y) for all y in Y
    """
    f_l =  np.zeros((n_shares, y_range)) # = [f(l1|Y), f(l2|Y), ..., f(l0|Y)]
    for i in range(n_shares):
        f_l[i] = f_li_given_Y(l[i], y_range, sigma)
    acc_f = np.zeros(y_range)
    for y in range(y_range):
        # print(f"====================={y}====================")
        # pretty_print_a(shares)
        all_shares = np.column_stack((shares, masked_Y[y]))
        prod_acc_f = np.ones(all_shares.shape[0])
        for i in range(n_shares):
            idx_share = all_shares[:, i]
            f_li = f_l[i]
            prod_acc_f *= f_li[idx_share]
            # pretty_print_a(idx_share)
            # pretty_print_a(f_li[idx_share])
            # print(prod_acc_f)
        acc_f[y] = prod_acc_f.sum()
    return acc_f

def p_y_given_l(l, y, shares, masked_Y, n_shares, y_range, sigma):
    """
    p(y|l) = f(l|y)/sum_Y(f(l|y'))
    """
    f_l_Y = f_l_given_Y(l, shares, masked_Y, n_shares, y_range, sigma)
    numerator =  f_l_Y[y]
    denominator = f_l_Y.sum()
    p = numerator/denominator
    return p

def conditional_entropy(shares, masked_Y, n_samples, n_shares, y_range, sigma):
    """
    1/(y_range)*1/n sum_y sum_l log2(P(y|l))
    """
    acc_p = 0
    for y in range(y_range):
        acc_py = 0
        y = np.array([y])
        print(f"Processing y = {y}, {HW(y)}")
        shares_ = gen_shares(y, n_shares=n_shares, n_samples=n_samples)
        leakages = gen_leakages(shares_, n_samples, sigma=sigma)
        L = np.array([val for val in leakages.values()]).T
        for l in L:
            acc_py += np.log2(p_y_given_l(l, y, shares, masked_Y, n_shares, y_range, sigma))
        acc_py = acc_py/n_samples
        acc_p += acc_py
    return acc_p/y_range

def MI(shares, masked_Y, n_samples, n_shares, y_range, sigma):
    con_ent = conditional_entropy(shares, masked_Y, n_samples, n_shares, y_range, sigma)
    return np.log2(y_range) + con_ent


if __name__ == '__main__':
    n_shares = 2
    y_range = 256
    # shares, masked_Y =  gen_all_shares_Y(y_range, n_shares)
    # with open("MIE/precomputed_shares.npy", "rb") as f:
    #     shares = np.load(f)
    #     masked_Y = np.load(f)
    # pretty_print_a(shares)
    # pretty_print_a(masked_Y)
    with open("precomputed_shares.npy", "rb") as f:
        shares = np.load(f)
        masked_Y = np.load(f)
    sigma_2 = np.linspace(-3, 1, 9, endpoint=True)
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    log_mie = np.linspace(0, -3, 7, endpoint=True)
    log_mie[:3] = 0

    I = np.zeros(sigma.shape)
    i = 0
    # print(sigma_2)
    # print(sigma)
    for s, m in zip(sigma, log_mie):
        n = int(2/10**m)
        print(f"========================={s}, {n}====================")
        mi = MI(shares, masked_Y, n, n_shares, y_range, s)
        print(mi)
        I[i] = np.log10(mi)
        i += 1

        with open("log.txt", "a") as f_:
            f_.write(f"{mi}\n")
    with open("MI.npy", "wb") as f:
        np.save(f, I)
    plt.scatter(sigma_2, I)
    plt.savefig("MI.png")
