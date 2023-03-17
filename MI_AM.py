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
    acc_f = np.zeros(S_size)
    for i, s in enumerate(s_range):
        all_shares = np.column_stack((shares, masked_S[i]))
        prod_acc_f = np.ones(all_shares.shape[0])
        for i_s in range(n_shares):
            idx_share = all_shares[:, i_s]
            f_li = f_l[i_s]
            prod_acc_f *= f_li[idx_share]
        acc_f[i] = prod_acc_f.sum()
    return acc_f

def p_s_given_l(l, s, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
    """Compute p(s|l)
        p(s|l) = f(l|s)/sum_s'(f(l|s'))

    Parameters
    ----------
    l: 1D array
        l = [l1, ..., ld-2, l0 = masked state]
    s: int
        Secret value
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
    p(s|l): float
            f(l|s)p(s)/sum_s'(f(l|s'))p(s')

    """
    f_l_S = f_l_given_S(l, s_range, shares, masked_S, n_shares, q, sigma)
    f_l_S = f_l_S*prior_s
    numerator =  f_l_S[s]
    denominator = f_l_S.sum()
    p = numerator/denominator
    return p


def conditional_entropy(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    """Estimate H(S|L) with n_samples
        =1/n sum_s sum_l p(s) log2(P(s|l))

    Parameters
    ----------
    s_range : array size S
        Possible values for s
    prior_s : array size S
        Prior proba of s (binomial distribution)
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
        s = np.array([s])
        print(f"Processing y = {s}, {HW(s)}")
        shares_ = gen_shares_am(s, q, n_shares, n_samples, seed, op, mode)
        leakages = gen_leakages(shares_, n_samples, sigma)
        L = np.array([val for val in leakages.values()]).T
        acc_ps = 0
        for l in L:
            psl = p_s_given_l(l, i, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
            acc_ps += np.log2(psl)
        acc_ps = acc_ps
        acc_p += acc_ps*prior_s[i]
    return acc_p/n_samples

def MI(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    S_size = s_range.shape[0]
    con_ent = conditional_entropy(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed, op, mode)
    return ent_s(prior_s) + con_ent

def ent_s(prior_s):
    log_2p = np.log2(prior_s)
    return -(prior_s*log_2p).sum()

def run_MI(q):
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
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
    prior_s = prior_ps()
    for s, m in zip(sigma, log_mie):
        rand_seed = 12345
        print(rand_seed)
        n = int(100000/10**m)
        print(f"=========================sigma^2: {np.log10(s**2)}, n_samples: {n}====================")
        print("==================OP SUB========================")
        shares, masked_S = load_precomputed_vals(q, n_shares, op="sub", mode="1")
        mi_sub = MI(s_range, prior_s, shares, masked_S, n, n_shares, q, s, seed=rand_seed)
        print("==================OP ADD========================")
        shares, masked_S = load_precomputed_vals(q, n_shares, op="add", mode="1")
        mi_add = MI(s_range, prior_s, shares, masked_S, n, n_shares, q, s, seed=rand_seed, op="add")
        with open(f"log/log_{q}_{n_shares}.txt", "a") as f_:
            f_.write(f"{mi_sub}_{mi_add}_{s}\n")
        mi_sub = np.log10(mi_sub)
        mi_add = np.log10(mi_add)
        I_sub[i] = mi_sub
        I_add[i] = mi_add
        i += 1

    with open(f"log/MI_{q}_{n_shares}_add.npy", "wb") as f:
        np.save(f, I_add)
    with open(f"log/MI_{q}_{n_shares}_sub.npy", "wb") as f:
        np.save(f, I_sub)
    plt.scatter(sigma_2, I_add, color="blue", marker="o", s=10)
    plt.plot(sigma_2, I_add, color="blue", label="s1 = (s + s0)%q")
    plt.scatter(sigma_2, I_sub, color="red", marker="x", s=10)
    plt.plot(sigma_2, I_sub, color="red", label="s1 = (s - s0)%q")
    plt.xlabel("$\log_{10}(\sigma^{2})$")
    plt.ylabel("$\log_{10}(MI)$")
    plt.legend()
    plt.title(f"q={q}")
    plt.savefig(f"pic/MI_{q}_{n_shares}.png")

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
    # print()
    # f_test()
    # shares, masked_S = gen_all_shares_S(s_range, q, n_shares)
    # shares, masked_S = gen_all_shares_S(s_range, q, n_shares, op="add", mode=1)
    run_MI(q=q)
    # print(ent_s(prior_ps()))
    # with open("log/MI_23_2_add.npy", "rb") as f:
    #     Iadd = np.load(f)
    # with open("log/MI_23_2_sub.npy", "rb") as f:
    #     Isub = np.load(f)
    # pretty_print_a(10**Iadd, tag="add")
    # pretty_print_a(10**Isub, tag="\nsub")
    # print(ent_s(prior_ps()))
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
