import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *



def f_li_given_Y(li, y_range, sigma):
    """
    f(l=li|y=y_i) with y_i runs over y_range
    return: [f(l=li|y=0), f(l=li|y=1), ..., f(l=li|y=255)]
    """
    y_range_ = np.arange(y_range)
    hw_set = HW(y_range_)
    hw_unique = np.unique(hw_set)
    li_ = np.repeat(li, repeats=hw_unique.shape[0], axis=0)
    pdf_hw = pdf_normal(li_, hw_unique, sigma)
    pdf_li = pdf_hw[hw_set]
    return pdf_li



def f_l_given_y(l, y, all_shares, n_shares, y_range, sigma):
    """
    f(L=[l0, ..., l_d-1]|Y=y)
    """
    y_range_ = np.arange(y_range)
    pdf_ = np.zeros((n_shares, y_range))
    # print(f"=================++Before first call {list(all_shares)}===============")
    for i in range(n_shares):
        pdf_[i] = f_li_given_Y(l[i], y_range, sigma)
    acc_f = 0
    if n_shares <= 1:
        for share in all_shares:
            acc_f += pdf_.squeeze()[y.squeeze()]
    else:
        for shares in all_shares:
            shares = np.array(shares)
            masked_y = np.bitwise_xor.reduce(shares)^y
            share_vals = np.insert(shares, 0, masked_y)
            acc_pdf = np.array([pdf_[val, i] for val, i in enumerate(share_vals)])
            acc_f += np.prod(acc_pdf)
            # print(f"=================++After first call {list(all_shares)}===============")
            # print("share_vals:", share_vals, HW(share_vals), np.prod(acc_pdf), acc_f)
            # print("pdf", acc_pdf, np.prod(acc_pdf), "acc_pdf", acc_f)
        # print("acc_f", acc_f)
        acc_f/(y_range**(n_shares-1))
        # print(acc_f)
    return acc_f
def f_l_given_y_ho(l, y, all_shares, n_shares, y_range, sigma):
    """
    f(L=[l0, ..., l_d-1]|Y=y)
    """
    y_range_ = np.arange(y_range)
    pdf_ = np.zeros((n_shares, y_range))
    for i in range(n_shares):
        pdf_[i] = f_li_given_Y(l[i], y_range, sigma)
    acc_f = 0
    if n_shares <= 1:
        for share in all_shares:
            acc_f += pdf_.squeeze()[y.squeeze()]
    else:
        for shares in all_shares:
            shares = np.array(shares)
            masked_y = np.bitwise_xor.reduce(shares)^y
            share_vals = np.insert(shares, 0, masked_y)
            acc_pdf = np.array([pdf_[val, i] for val, i in enumerate(share_vals)])
            acc_f += np.prod(acc_pdf)
            # print("share_vals:", share_vals, HW(share_vals), np.prod(acc_pdf), acc_f)
            # print("pdf", acc_pdf, np.prod(acc_pdf), "acc_pdf", acc_f)
        # print("acc_f", acc_f)
        acc_f/(y_range**(n_shares-1))
        # print(acc_f)
    return acc_f
def f_l_given_Y_ho(l, y, all_shares, n_shares, y_range, sigma, f="YYY"):
    """
    sum f(L=[l0, ..., l_d-1]|Y=y) for all y in y_range
    """
    acc_f = 0
    for y_ in range(y_range):
        y_ = np.array([y_])
        f_ = f_l_given_y_ho(l, y_, all_shares, n_shares, y_range, sigma)
        acc_f += f_
    return acc_f

def f_l_given_Y(l, all_shares, n_shares, y_range, sigma, f="YYY"):
    """
    sum f(L=[l0, ..., l_d-1]|Y=y) for all y in y_range
    """
    acc_f = 0
    for y_ in range(y_range):
        y_ = np.array([y_])
        f_ = f_l_given_y(l, y_, all_shares, n_shares, y_range, sigma)
        acc_f += f_
    return acc_f

def p_y_given_l(l, y, n_shares, y_range, sigma):
    all_shares = gen_all_shares(y, n_shares, 256)
    numerator =  f_l_given_y(l, y, all_shares, n_shares, y_range, sigma)
    if n_shares <= 1:
        denominator = f_l_given_Y(l, all_shares, n_shares, y_range, sigma)
    else:
        denominator = f_l_given_Y_ho(l, y, all_shares, n_shares, y_range, sigma)

    p = numerator/denominator
    return p


def conditional_entropy(n_samples, n_shares, y_range, sigma):
    """
    1/(y_range)*1/n sum_y sum_l log2(P(y|l))
    """
    acc_p = 0
    for y in range(y_range):
        acc_py = 0
        y = np.array([y])
        print(f"Processing y = {y}, {HW(y)}")
        shares = gen_shares(y, n_shares=n_shares, n_samples=n_samples)
        leakages = gen_leakages(shares, n_samples, sigma=sigma)
        L = np.array([val for val in leakages.values()]).T
        for l in L:
            acc_py += np.log2(p_y_given_l(l, y, n_shares, y_range, sigma))
        acc_py = acc_py/n_samples
        acc_p += acc_py
    return acc_p/y_range



def MI(n_samples, n_shares, y_range, sigma):
    con_ent = conditional_entropy(n_samples, n_shares, y_range, sigma)
    return np.log2(y_range) + con_ent

def f(x, s, sigma):
    first_comp = np.exp(-(x-HW(s))**2/(2*(sigma**2)))
    numerator = 0
    for s_ in range(256):
        numerator += np.exp(-(x-HW(s_))**2/(2*(sigma**2)))
    second_comp = np.log2(numerator) - np.log2(first_comp)
    return second_comp*first_comp

def MI_itegral():
    acc_i = 0
    for y in range(256):
        x = np.linspace(-100, 100, num=10000, endpoint=True)
        s = y
        acc_i += np.trapz(f(x, s, 10), x=x)
    q = 1/(np.sqrt(2*np.pi))
    q = q/10
    print(8 - acc_i*q/256)


if __name__ == '__main__':
    # n_samples = 1
    # n_shares = 3
    # # y = np.random.randint(0, 256, (n_samples, ))
    # y = np.array([6])
    # print(y)
    # shares = gen_shares(y, n_shares=n_shares, n_samples=n_samples)
    # [print(f"{key} {val} {HW(val)}") for key, val in shares.items()]
    # print(y)
    # leakages = gen_leakages(shares, n_samples, sigma=0.1)
    # [print(f"{key} {val}") for key, val in leakages.items()]
    # print(y)
    # all_shares = gen_all_shares(y, n_shares, y_range=256)
    # for shares in all_shares:
    #     shares = np.array(shares)
    #     masked_y = np.bitwise_xor.reduce(shares)^y
    #     share_vals = np.append(shares, masked_y)
    #     print(share_vals)
    # n_samples = 2
    n_shares = 2
    y_range = 256
    sigma_2 = np.linspace(-3, 1, 9, endpoint=True)
    print(sigma_2)
    sigma_2 = np.power(10, sigma_2)
    sigma = np.sqrt(sigma_2)
    log_mie = np.linspace(0, -3, 7, endpoint=True)
    log_mie[:3] = 0

    # print(log_mie)
    # y = np.array([15])
    # print(sigma)
    # print(10**(-3), 1/10**(-3))
    # print(norm.pdf(0, 0, sigma))
    # print(pdf_normal(0, 0, sigma))
    I = np.zeros(sigma.shape)
    i = 0
    print(sigma_2)
    print(sigma)
    for s, m in zip(sigma, log_mie):
        n = int(2/10**m)
        print(f"========================={s}, {n}====================")
        mi = MI(n, n_shares, y_range, s)
        print(mi)
        I[i] = np.log10(mi)
        i += 1
        plt.scatter(sigma_2, I)
        plt.savefig("MI.png")
        with open("log.txt", "a") as f_:
            f_.write(f"{mi}\n")
    with open("MI.npy", "wb") as f:
        np.save(f, I)




    # print(MI(n_samples, n_shares, y_range, sigma))
    # MI_itegral()
    # print(HW(y))
    # shares = gen_shares(y, n_shares=n_shares, n_samples=n_samples)
    # pretty_print_dict(shares)
    # leakages = gen_leakages(shares, n_samples, sigma=var)
    # pretty_print_dict(leakages)
    # L = np.array([val for val in leakages.values()]).T
    # for l in L:
    #     print(l.shape, l)
    # conditional_entropy(n_samples=1, n_shares=1, y_range=256, var=0.1)
    # f_li_given_Y(leakages['Y'][0], 255, 0.1)
    # print(f_li_given_Y(leakages['X_0'][0], 255, 0.1))
    # l = np.array([val for val in leakages.values() ])
    # print(l)
    # # f_li_given_y(leakages['X_0'][0], )
    # f_l_given_y(l, y, n_shares, y_range=256, var=0.1)
    # print(it.combinations(p, 2))
    # for comb in it.combinations_with_replacement(p, 2):
    #     c = np.array(comb)
    #     print(c)
        # print(c, np.bitwise_xor.reduce(c))
    # print(x_1, y_1)
    # re = np.array(np.meshgrid([p1, p2]))
    # print(re.shape)
    # re = re.T
    # print(re.reshape(-1,2))
    # arrays = [np.fromiter(range(100), dtype=int), np.fromiter(range(100, 200), dtype=int)]
    # arrayz = np.array([p1, p2])
    # itera = it.product(*p_)
    # for c in itera:
    #     print(c)
    # n_samples = 1
    # n_shares = 3
    # y = np.array([91])
    # all_shares = gen_all_shares(y, n_shares, 256)
    # shares = gen_shares(y, n_shares=n_shares, n_samples=n_samples)
    # [print(f"{key} {val} {HW(val)}") for key, val in shares.items()]
    # leakages = gen_leakages(shares, n_samples, sigma=0.1)
    # [print(f"{key} {val}") for key, val in leakages.items()]
    # l = np.array([val for val in leakages.values() ])
    # f_l_given_y(l, y, all_shares, n_shares, y_range=256, var=0.1)
