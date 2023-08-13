import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

def construct_exact_pre_knowledge(nnodes_1, nnodes_2):
    Douban = loadmat('datasets/Douban/Douban.mat')
    H = Douban['H']
    h = np.zeros(nnodes_1 * nnodes_2)
    rcv = sp.find(H)
    for line in range(len(rcv[0])):
        node_in_sub = rcv[0][line]
        node_in_whole = rcv[1][line]
        sim = rcv[2][line]
        h[node_in_whole * nnodes_2 + node_in_sub] = sim
    return h

def construct_pre_knowledge(nnodes1, nnodes2):
    h = np.zeros(nnodes1 * nnodes2)
    pre = 1/(nnodes1*nnodes2)
    for i in range(nnodes1 * nnodes2):
        h[i] = pre
    return h