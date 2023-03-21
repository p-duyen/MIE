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
def p_s_given_l_(l, s, s_range, prior_s, shares, masked_S, n_shares, q, sigma):
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
    numerator_ =  f_l_S[s]
    f_l_S = f_l_S*prior_s
    numerator =  f_l_S[s]
    denominator = f_l_S.sum()
    if denominator == 0:
        p = 0
    else:
        p = numerator/denominator
    return numerator_, p


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
        acc_p += acc_ps*prior_s[i]
    return acc_p/n_samples
def conditional_entropy_(s_range, prior_s, shares, masked_S, n_shares, q, sigma, seed=12345, op="sub", mode=1):
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
    acc_h = 0
    n_points = 400
    lmin = -1
    lmax = 5
    int_l0 = np.linspace(lmin, lmax, n_points)
    int_l1 = np.linspace(lmin, lmax, n_points)
    int_l = np.meshgrid(int_l0, int_l1, indexing="ij")
    g = np.zeros((n_points, n_points))
    for i_ in range(n_points):
        for j_ in range(n_points):
            l = np.array([int_l[0][i_, j_], int_l[1][i_, j_]])
            for i, s in enumerate(s_range):
                fls, psl = p_s_given_l_(l, i, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
                if psl == 0:
                    g[i_, j_]+= -100000*fls*prior_s[i]
                else:
                    g[i_, j_] += fls*np.log2(psl)*prior_s[i]
    h = simpson(simpson(g, int_l1), int_l0)/q
    return h

def conditional_entropy__(s_range, prior_s, shares, masked_S, n_shares, q, sigma, seed=12345, op="sub", mode=1):
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
    acc_h = 0
    n_points = 400
    lmin = -1
    lmax = 5
    int_l0 = np.linspace(lmin, lmax, n_points)
    int_l1 = np.linspace(lmin, lmax, n_points)
    int_l = np.meshgrid(int_l0, int_l1, indexing="ij")
    g = np.zeros((n_points, n_points))
    for i, s in enumerate(s_range):
        print(f"Processing y = {s}, {HW(s)} i = {i}")
        for i_ in range(n_points):
            for j_ in range(n_points):
                l = np.array([int_l[0][i_, j_], int_l[1][i_, j_]])
                fls, psl = p_s_given_l_(l, i, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
                if psl == 0:
                    g[i_, j_]+= -100000*fls*prior_s[i]
                else:
                    g[i_, j_] += fls*np.log2(psl)*prior_s[i]
    h = simpson(simpson(g, int_l1), int_l0)/q
    return h

def cond_ent_compose(s, i, acc_p, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    s = np.array([s])
    print(f"Processing y = {s}, {HW(s)} i = {i}")
    shares_ = gen_shares_am(s, q, n_shares, n_samples, seed, op, mode)
    leakages = gen_leakages(shares_, n_samples, sigma)
    L = np.array([val for val in leakages.values()]).T
    acc_ps = 0
    for l in L:
        psl = p_s_given_l(l, i, s_range, prior_s, shares, masked_S, n_shares, q, sigma)
        acc_ps += np.log2(psl)
    acc_p[i] += acc_ps*prior_s[i]

def cond_ent_multiprocess(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    manager = Manager()
    acc_p = manager.list([0, 0, 0, 0, 0])
    idx = np.arange(s_range.shape[0])
    f_ = partial(cond_ent_compose, acc_p=acc_p, prior_s=prior_s, shares=shares, masked_S=masked_S, n_samples=n_samples, n_shares=n_shares, q=q, sigma=sigma, seed=seed, op=op, mode=mode)
    pool = Pool(s_range.shape[0])
    pool.starmap(f_, zip(s_range, idx))
    pool.close()
    pool.join()
    acc_p_l = np.array(acc_p)
    return acc_p_l.sum()/n_samples

def MI_(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    S_size = s_range.shape[0]
    con_ent = conditional_entropy(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed, op, mode)
    return ent_s(prior_s) + con_ent
def MI__(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    S_size = s_range.shape[0]
    con_ent = conditional_entropy__(s_range, prior_s, shares, masked_S, n_shares, q, sigma, seed, op, mode)
    return ent_s(prior_s) + con_ent

def MI(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1):
    S_size = s_range.shape[0]
    con_ent = cond_ent_multiprocess(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed, op, mode)
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
    shares_sub, masked_S_sub = load_precomputed_vals(q, n_shares, op="sub", mode="1")
    shares_add, masked_S_add = load_precomputed_vals(q, n_shares, op="add", mode="1")
    for s, m in zip(sigma, log_mie):
        rand_seed = 12345
        n = int(100000/10**m)
        print(f"=========================sigma^2: {np.log10(s**2)}, n_samples: {n}====================")
        print("==================OP SUB========================")
        mi_sub = MI(s_range, prior_s, shares_sub, masked_S_sub, n, n_shares, q, s, seed=rand_seed)
        print("==================OP ADD========================")
        mi_add = MI(s_range, prior_s, shares_add, masked_S_add, n, n_shares, q, s, seed=rand_seed, op="add")
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

def test_f():
    q = 23
    # run_MI(q=19)
    n_shares = 2
    sigma = 0.03162277660168379
    s_range = np.array([-2, -1, 0, 1, 2])
    n_samples = 4000
    prior_s = prior_ps()
    # f_Li_gievn_S(Li, s_range, shares, masked_S, n_shares, q, sigma)
    s = np.array([s_range[2]])
    # shares = gen_shares_am(s, q, n_shares, n_samples)
    # leakages = gen_leakages(shares, n_samples, sigma)
    # L = np.array([val for val in leakages.values()])
    op = "add"
    shares, masked_S = load_precomputed_vals(q, n_shares, op=op, mode="1")
    # print(conditional_entropy_(s_range, prior_s, shares, masked_S, n_shares, q, sigma, seed=12345, op="sub", mode=1))
    print(MI_(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op=op, mode=1))
    print("===========================")
    print(MI__(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op=op, mode=1))
    # print(conditional_entropy(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1))
    # print("================\n",L.shape, L)
    # print(L.reshape(50, 50))
    # pretty_print_dict(shares)

    # pretty_print_dict_HW(shares)
    # pretty_print_dict(leakages)
    # f_L = f_L_given_S(L, s_range, shares, masked_S, n_shares, q, sigma)
    # # # g_s_given_L(s, f_L, prior_s, n_shares, q, sigma, seed=12345, op="sub", mode=1)
    # conditional_entropy_(s_range, prior_s, shares, masked_S, n_samples, n_shares, q, sigma, seed=12345, op="sub", mode=1)

def tiny_test():
    lmin = -2
    lmax = 4
    n_points = 100
    int_l0 = np.linspace(lmin, lmax, n_points)
    int_l1 = np.linspace(lmin, lmax, n_points)
    L = np.meshgrid(int_l0, int_l1, indexing="ij")
    print(L)

if __name__ == '__main__':
    q = 1009
    n_shares = 2
    s_range = np.array([-2, -1, 0, 1, 2])
    # print(np.linspace(-5, 20, num=200, endpoint=True))
    # L = int_l()
    #
    # print(L[0].shape, L[1].shape)
    # f_test()
    # test_f()
    tiny_test()
    # x = np.linspace(0, 1, 20)
    # y = np.linspace(0, 1, 30)
    # z = np.cos(x[:,None])**4 + np.sin(y)**2
    # print(z.shape, z)
    # print(simps(z, y).shape)
    # print(simps(simps(z, y), x))
    # shares, masked_S = gen_all_shares_S(s_range, q, n_shares)
    # shares, masked_S = gen_all_shares_S(s_range, q, n_shares, op="add", mode=1)
    # run_MI(q=q)
    # print(ent_s(prior_ps()))
    # with open("log/MI_23_2_add.npy", "rb") as f:
    #     Iadd = np.load(f)
    # with open("log/MI_23_2_sub.npy", "rb") as f:
    #     Isub = np.load(f)
    # pretty_print_a(10**Iadd, tag="add")
    # pretty_print_a(10**Isub, tag="\nsub")
