import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *
from time import time



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

def f_l_given_S(l, s_range, shares, masked_S, n_shares, q, sigma):
    """Compute f(l|s) for all possible value of the secret s

    Parameters
    ----------
    l: 1D array
        l = [l1, ..., ld-2, l0 = masked state]
    s_range : array size S
        Possible values for s
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
    f(l|S): array size S
            f(l|S)[i] = f(l|si)

    """
    S_size = s_range.shape[0]
    f_l =  np.zeros((n_shares, q)) # = [f(l1|S), f(l2|S), ..., f(l0|S)]
    for i in range(n_shares):
        f_l[i] = f_li_given_S(l[i], q, sigma) # f(li|S) S in Zq
        # pretty_print_a(f_l[i], tag=f"\nshare {i}")
    acc_f = np.zeros(S_size)
    for i, s in enumerate(s_range):
        # print(f"\n=============Computing for {s}=================")
        all_shares = np.column_stack((shares, masked_S[i]))
        # pretty_print_a(shares, tag=f"shares for {s}")
        # pretty_print_a(masked_S[i], tag=f"\nmasked for {s}")
        prod_acc_f = np.ones(all_shares.shape[0])
        for i_s in range(n_shares):
            idx_share = all_shares[:, i_s]
            f_li = f_l[i_s]
            prod_acc_f *= f_li[idx_share]
            # print(f"\n=================share {i_s}================")
            # pretty_print_a(idx_share, tag=f"chosen idx")
            # pretty_print_a(f_li[idx_share], tag=f"chosen pdf")
        acc_f[i] = prod_acc_f.sum()
    return acc_f

def p_s_given_l(l, s, s_range, shares, masked_S, n_shares, q, sigma):
    """Compute p(s|l) for all possible value of the secret s
        p(s|l) = f(l|s)/sum_s'(f(l|s'))

    Parameters
    ----------
    l: 1D array
        l = [l1, ..., ld-2, l0 = masked state]
    s: int
        Secret value
    s_range : array size S
        Possible values for s
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
    f(l|S): array size S
            f(l|S)[i] = f(l|si)

    """
    f_l_S = f_l_given_S(l, s_range, shares, masked_S, n_shares, q, sigma)
    numerator =  f_l_S[s]
    denominator = f_l_S.sum()
    p = numerator/denominator
    # print(f"======Compute p(s= {s_range[s]}|l = {l}) {numerator}/{denominator} = {p}==========")
    return p


def conditional_entropy(s_range, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    """Estimate H(S|L) with n_samples
        = 1/(q)*1/n sum_s sum_l log2(P(s|l))

    Parameters
    ----------
    s_range : array size S
        Possible values for s
    shares: array shape: (n_shares-1, q^(n_shares-1))
        Precomputed all possible randomness (l1, ..., ld-2)
    masked_S: array shape: (S, q^(n_shares-1))
        Precomputed masked secret (l0) for all possible value of secret
    n_samples: int
        Number of samples
    n_shares: int
        Number of shares
    q: int
        Prime
    sigma: float
        Standard deviation for Gaussian leakages
    seed: int
        Seed for prg, force different op and mode to have the same shares values
    op: string "add" or "sub"
        Operations to compute masked s (s + (q - sum_mod(shares)) for "add", s - sum_mod(shares) for "sub"))
    mode: int 0, 1
        For add op: 0 to convert every shares to the additive inverse, 1 to convert 1 share to additive inverse

    Returns
    -------
    Estimate H(S|L)

    """
    acc_p = 0
    for i, s in enumerate(s_range):
        acc_ps = 0
        s = np.array([s])
        print(f"Processing y = {s}, {HW(s)}")
        shares_ = gen_shares_am(s, q, n_shares, n_samples, seed, op, mode)
        # pretty_print_dict(shares_, tag="shares")
        leakages = gen_leakages(shares_, n_samples, sigma)
        # pretty_print_dict(leakages, tag="leakage")
        L = np.array([val for val in leakages.values()]).T
        for l in L:
            psl = p_s_given_l(l, i, s_range, shares, masked_S, n_shares, q, sigma)
            acc_ps += np.log2(psl)
        acc_ps = acc_ps/n_samples
        acc_p += acc_ps
    return acc_p/(s_range.shape[0])

def MI(s_range, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    S_size = s_range.shape[0]
    con_ent = conditional_entropy(s_range, shares, masked_S, n_samples, n_shares, q, sigma, seed, op, mode)
    return np.log2(S_size) + con_ent

def f_test():
    q = 17
    n_shares = 2
    n_samples = 5
    s_range = np.array([-2, -1, 0, 1, 2])
    s = s_range[0]
    sigma = 0.1
    shares = gen_shares_am(s, q, n_shares, n_samples)
    leakages = gen_leakages(shares, n_samples, sigma)
    L = np.array([val for val in leakages.values()]).T
    # pretty_print_dict(shares, tag=f"secret{s}")
    # pretty_print_dict_HW(shares,  tag=f"secret HW {HW(s%q)}")
    # pretty_print_dict(leakages, tag="leakages")
    # #=========================test f_li_given_S===================
    # L1 = L[0]
    # f_l = f_li_given_S(L1[0], q, sigma)
    # s_1 = shares["S_1"][0]
    # print(HW(np.arange(q)))
    # print(f"share_val: {s_1}, leakage: {L1[0]}, pdf:\n{f_l}\n{f_l.shape}")
    # gen_all_shares_S(s_range, q, n_shares)
    #======================test f_l_given_S======================
    with open("precomputed_shares_am_17_2shares_sub_mode1.npy", "rb") as f:
        shares = np.load(f)
        masked_S = np.load(f)
    l = L[0]
    # f_l = f_l_given_S(l, s_range, shares, masked_S, n_shares, q, sigma)
    # print(f"leakage: {l}, pdf:\n{f_l}\n{f_l.shape}")
    # print("\n")
    # for s in s_range()
    # print(p_s_given_l(l, s, s_range, shares, masked_S, n_shares, q, sigma))
    # print(conditional_entropy(s_range, shares, masked_S, n_samples, n_shares, q, sigma))
    print(MI(s_range, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1))


def unitest():
    q = 3329
    n_shares = 2
    s_range_o = np.array([-2, -1, 0, 1, 2])
    s_range = s_range_o%q
    s = s_range[0]
    shares = gen_shares_am(s, q, n_shares, 3)
    pretty_print_dict(shares, tag=f"{s}")
    pretty_print_dict_HW(shares,  tag=f"HW{s}")
    leakages = gen_leakages(shares, 1, 0.1)
    pretty_print_dict(leakages, tag="leakages")
    shares_ = gen_shares_am(s, q, n_shares, 3, op="add")
    pretty_print_dict(shares_, tag=f"{s}_add_1")
    shares__ = gen_shares_am(s, q, n_shares, 3, op="add", mode=0)
    pretty_print_dict(shares__, tag=f"{s}_add_0")


def run_MI(q):
    # manager = plt.get_current_fig_manager()
    # manager.frame.Maximize(True)
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    sparse_log_mie = np.zeros_like(sparse_sigma_2)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    dense_log_mie = np.linspace(-0.1, -2.75, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    log_mie = np.hstack((sparse_log_mie, dense_log_mie))
    # [print(sigma_2[i], log_mie[i]) for i in range(log_mie.shape[0])]
    sigma_2_10 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2_10)
    I_sub = np.zeros_like(sigma_2)
    I_add = np.zeros_like(sigma_2)
    i = 0
    for s, m in zip(sigma, log_mie):
        n = int(100/10**m)
        print(f"=========================sigma^2: {np.log10(s**2)}, n_samples: {n}====================")
        print("==================OP SUB========================")
        shares, masked_S = load_precomputed_vals(q, n_shares, op="sub", mode="1")
        mi_sub = MI(s_range, shares, masked_S, n, n_shares, q, s, seed=12345)
        print("==================OP ADD========================")
        shares, masked_S = load_precomputed_vals(q, n_shares, op="add", mode="1")
        # print(shares.shape, masked_S.shape)
        mi_add = MI(s_range, shares, masked_S, n, n_shares, q, s, seed=12345, op="add")
        mi_sub = np.log10(mi_sub)
        mi_add = np.log10(mi_add)
        I_sub[i] = mi_sub
        I_add[i] = mi_add
        i += 1
        with open(f"log_{q}_{n_shares}.txt", "a") as f_:
            f_.write(f"{mi_sub}_{mi_add}\n")
    with open(f"MI_{q}_{n_shares}_add.npy", "wb") as f:
        np.save(f, I_add)
    with open(f"MI_{q}_{n_shares}_sub.npy", "wb") as f:
        np.save(f, I_sub)
    plt.scatter(sigma_2, I_add, color="blue", marker="o", s=10)
    plt.plot(sigma_2, I_add, color="blue", label="s1 = (s + s0)%q")
    plt.scatter(sigma_2, I_sub, color="red", marker="x", s=10)
    plt.plot(sigma_2, I_sub, color="red", label="s1 = (s - s0)%q")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"q={q}")
    plt.savefig(f"MI_{q}_{n_shares}.png")

def check_add():
    sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    dense_sigma_2 = np.linspace(-1.25, 1, 19)
    sigma_2 = np.hstack((sparse_sigma_2, dense_sigma_2))
    qs = [5, 11, 13, 17, 19]
    for Q in qs:
        with open(f"MI_{Q}_2_add.npy", "rb") as f:
            mi = np.load(f)
        plt.plot(sigma_2, mi, label=f"q={Q}")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"s1 = (s + s0)%q")
    plt.savefig(f"MI_add_{qs}.png")

    return 0
if __name__ == '__main__':
    q = 23
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    # f_test()
    shares, masked_S = gen_all_shares_S(s_range, q, n_shares)
    shares, masked_S = gen_all_shares_S(s_range, q, n_shares, op="add", mode=1)
    run_MI(q=q)
    # check_add()
    # sparse_sigma_2 = np.linspace(-3, -1.5, 4)
    # sparse_log_mie = np.zeros_like(sparse_sigma_2)
    # dense_log_mie = np.linspace(-0.1, -2.75, 19)
    # log_mie = np.hstack((sparse_log_mie, dense_log_mie))
    # for m in log_mie:
    #     print(int(100/10**m), end=" ")
    # s_range = s_range_o%q
    # s = s_range[0]
    # st = time()
    # print(time() - st)
    # print(shares.shape)
    # print(masked_S.shape)
    # print(shares[:3])
    # print(masked_S[:, :3])
