import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import itertools as it
from scipy.integrate import quad
from scipy.stats import norm
from utils_ import *
from time import time
from multiprocessing import Pool, Manager, Value, Array, Process
from functools import partial
import multiprocessing.pool
import math
import logging

from tqdm import tqdm

import gc

logging.basicConfig(level=logging.INFO, filename='logging_MI_sampling', encoding='utf-8')

class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc

def draw_shares(secrets, n_traces, q, n_shares, op):
    shares = {}
    sum_acc = np.zeros((n_traces, ))
    for i in range(n_shares-1):
        tmp = np.random.randint(0, q, (n_traces, ), dtype=np.int16)
        shares[f"S{i}"] = tmp if op=="sub" else (q-tmp)%q
        sum_acc += tmp
    shares[f"S{n_shares-1}"] = (secrets - sum_acc)%q
    return shares, sum_acc
def draw_leakages(shares, sigma):
    L = {}
    for share, share_vals in shares.items():
        L[share] = HW(share_vals) + np.random.normal(0, sigma, size=share_vals.shape)
    return L
def pdf_X_given_l(l, q, sigma=0.1):
    Zq = np.arange(q)
    hw_set = HW(Zq)
    pdf = pdf_normal(l, hw_set, sigma)
    return pdf/pdf.sum(axis=1, keepdims=True)

def p_s_given_L(s, L, sigma, shares, n_shares, q):
    Zq = np.arange(q)
    hw_set = HW(Zq)
    shares_vals, acc_val = shares
    n_traces = acc_val.shape[0]
    p_Xi_given_Li = np.ones((n_traces, q))
    masked_s = (s - acc_val)%q
    for i in range(n_shares-1):
        Xi = shares_vals[i]
        f_Li_given_Xi = np.apply_along_axis(pdf_normal, 0, np.expand_dims(L[i], 0), mu=hw_set, sigma=sigma).T
        p_Xi_given_Li *= (f_Li_given_Xi/(f_Li_given_Xi.sum(axis=1, keepdims=True)))
    f_Lms_given_MS = np.apply_along_axis(pdf_normal, 0, np.expand_dims(L[n_shares-1], 0), mu=HW((s-Zq)%q), sigma=sigma).T
    p_Xi_given_Li *= (f_Lms_given_MS/(f_Lms_given_MS.sum(axis=1, keepdims=True)))
    return p_Xi_given_Li.sum(axis=1)


def MI_worker(sigma, n_traces, n_shares, q, op):
    print(f"Process {sigma}")
    s_range = [-2, -1, 0, 1, 2]
    secrets = np.random.choice(s_range, n_traces, p=prior_ps())
    shares = draw_shares(secrets, n_traces, q, n_shares, op)
    L = draw_leakages(shares[0], sigma)
    p = []
    for s in s_range:
        p.append(p_s_given_L(s, L, sigma, shares, n_shares, q))
    res = np.array(p)
    ent_S = ent_s(prior_ps())
    return ent_S - np.nansum(-(np.log2(res) * res), axis=0).mean()


def MI_compute(q, n_shares):
    sigma_2, sigma = gen_sigma()
    n_traces = 100000
    op = "sub"
    MI_f = partial( MI_worker, n_traces=n_traces, n_shares=n_shares, q=q, op=op)
    mi_holder = []
    with Pool(5) as pool:
        res = pool.map_async(MI_f, sigma)
        for val in res.get():

            mi_holder.append(np.log10(val))
    plt.plot(sigma_2, mi_holder)
    plt.scatter(sigma_2, mi_holder)
    plt.show()

if __name__ == '__main__':
    MI_compute(23, 2)
    import sys
    # print(sys.argv[1])
    # MI_worker(0.1, 10, 2, 23, "sub")
