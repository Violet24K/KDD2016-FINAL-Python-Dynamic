import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import math
import time
import pdb

def greedy_match(s: np.ndarray, nnodes_1, nnodes_2):
    min_size = min(nnodes_1, nnodes_2)
    used_rows = np.zeros(nnodes_2)
    used_cols = np.zeros(nnodes_1)
    exp_match = np.zeros(min_size)
    row = np.zeros(min_size)
    col = np.zeros(min_size)
    y = -np.sort(-s)
    ix = np.argsort(-s)
    matched = 0
    index = 0

    while(matched < min_size):
        ipos = ix[index]
        jc = math.floor(ipos/nnodes_2)
        ic = ipos-jc*nnodes_2
        if (used_rows[ic] != 1) and (used_cols[jc] != 1):
            row[matched] = ic
            col[matched] = jc
            exp_match[ic] = jc
            used_rows[ic] = 1
            used_cols[jc] = 1
            matched += 1
        index += 1
    return exp_match

# imputs are from sub to whole
def check_greedy_hit1(exp_match, actual_match, n_sub_node):
    hit = 0
    for i in range(n_sub_node):
        if (exp_match[i] == actual_match[i]):
            hit += 1
    hit_rate1 = hit/n_sub_node
    print('greedy hit rate1:', hit_rate1)
    return hit_rate1


def get_similarity(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, 
nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, 
node_attr_matrix_acce_1, node_attr_matrix_acce_2, h, alpha):
    print("constructing E matrix")
    E_1_L = {}
    E_2_L = {}
    for edge_attr_index in range(len(edge_attr_list)):
        edge_attr = edge_attr_list[edge_attr_index]
        E_1_l_matrix = sp.lil_matrix((nnodes_1, nnodes_1))
        for row_index in range(len(adj_matrix_1.rows)):
            for column_index in adj_matrix_1.rows[row_index]:
                if (edge_attr_matrix_1[row_index, column_index] == edge_attr):
                    E_1_l_matrix[row_index, column_index] = 1
        E_1_l_matrix = E_1_l_matrix.tocsr()
        E_1_L[edge_attr_index] = E_1_l_matrix
        E_2_l_matrix = sp.lil_matrix((nnodes_2, nnodes_2))
        for row_index in range(len(adj_matrix_2.rows)):
            for column_index in adj_matrix_2.rows[row_index]:
                if (edge_attr_matrix_2[row_index, column_index] == edge_attr):
                    E_2_l_matrix[row_index, column_index] = 1
        E_2_l_matrix = E_2_l_matrix.tocsr()
        E_2_L[edge_attr_index] = E_2_l_matrix

    K = node_attr_matrix_acce_1.shape[1]
    L = len(edge_attr_list)
    N = sp.csr_matrix((nnodes_1 * nnodes_2, 1))
    for k in range(K):
        N = N + sp.kron(node_attr_matrix_acce_1[:, k], node_attr_matrix_acce_2[:, k])
    N = N.toarray().ravel()

    print("constructing d")
    d = sp.csr_matrix((nnodes_1*nnodes_2, 1))
    for k in range(len(node_attr_list)):
        for l in range(len(edge_attr_list)):
            first_term = (E_1_L[l].multiply(adj_matrix_1)).dot(node_attr_matrix_acce_1[:, node_attr_list[k]])
            second_term = (E_2_L[l].multiply(adj_matrix_2)).dot(node_attr_matrix_acce_2[:, node_attr_list[k]])
            d += sp.kron(first_term, second_term, "csr")
    dd = np.zeros(nnodes_1 * nnodes_2)
    rcv = sp.find(d)
    for line in range(len(rcv[0])):
        if rcv[2][line] != 0:
            dd[rcv[0][line]] = (rcv[2][line])**(-1/2)

    # np.save("dd.npy", dd)
    # dd = np.load("dd.npy")
    start_time = time.time()

    q = np.multiply(dd, N)
    s = h.copy()

    for i in range(30):
        # print("iterate:", i)
        M_array = shape_an_array_from_vec(np.multiply(q, s), nnodes_2, nnodes_1)
        M = get_sparse_matrix_from_array(M_array)
        S = sp.csr_matrix((nnodes_2, nnodes_1))
        for l in range(L):
            S = S + (E_2_L[l].multiply(adj_matrix_2)).dot(M).dot((E_1_L[l].multiply(adj_matrix_1)).transpose())
        s = (1-alpha) * h +  alpha * np.multiply(q, vectorize_sparse_matrix(S))

    end_time = time.time()
    time_consumption = end_time - start_time
    print('time consumption: ', time_consumption)

    return s, time_consumption


def vectorize_sparse_matrix(S):
    numrows = S.shape[0]
    numcols = S.shape[1]
    v = np.zeros(numrows*numcols)
    rvc = sp.find(S)
    for line in range(len(rvc[0])):
        row = rvc[0][line]
        col = rvc[1][line]
        value = rvc[2][line]
        v[col * numrows + row] = value
    return v


def shape_an_array_from_vec(vec, shape1, shape2):
    if vec.shape[0] != shape1 * shape2:
        raise ValueError("vec should have shape1 * shape2 elements")
    M = np.zeros((shape1, shape2))
    for i in range(len(vec)):
        col = math.floor(i/shape1)
        row = i - col * shape1
        M[row][col] = vec[i]
    return M


def get_sparse_matrix_from_array(M):
    numrows = M.shape[0]
    numcols = M.shape[1]
    M_sparse = sp.lil_matrix((numrows, numcols))
    for i in range(numrows):
        for j in range(numcols):
            if M[i][j] != 0:
                M_sparse[i, j] = M[i][j]
    return M_sparse.tocsr()
