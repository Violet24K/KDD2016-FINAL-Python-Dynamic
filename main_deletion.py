import numpy as np
import scipy.sparse as sp
import construct
import pdb
import solve
DATASET_NAME = 'bitcoinalpha'
if DATASET_NAME == 'movielens-1m':
    from dataprocessing_temporal_movielens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'bitcoinalpha':
    from dataprocessing_temporal_bitcoinalpha import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'wikilens':
    from dataprocessing_temporal_wikilens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
import time

HAVE_NODE_ATTR = 1
HAVE_EDGE_ATTR = 1
GAP = 1

def get_graph_info(graph_file_name, node_attr_file_name):
    node_attr_set = set()
    edge_attr_set = set()
    graph_file = open(graph_file_name)
    node_attr_file = open(node_attr_file_name)
    for line in graph_file.readlines():
        items = line.strip().split(',')
        edge_attr = int(items[2])
        edge_attr_set.add(edge_attr)
    for line in node_attr_file.readlines():
        items = line.strip().split(',')
        node_attr = int(items[1])
        node_attr_set.add(node_attr)
    graph_file.close()
    node_attr_file.close()
    node_attr_list = []
    edge_attr_list = []
    for node_attr in node_attr_set:
        node_attr_list.append(node_attr)
    for edge_attr in edge_attr_set:
        edge_attr_list.append(edge_attr)
    return len(node_attr_set), len(edge_attr_set), node_attr_list, edge_attr_list


def build_matrices_accelerated(graph_file_name, node_attr_file_name, num_node_attr):
    graph_file = graph_file_name
    nodes_set = set()
    node_attr_file_open = open(node_attr_file_name, 'r')
    # first going through to count number of nodes
    for line in node_attr_file_open.readlines():
        items = line.strip().split(',')
        node = int(items[0])
        attr = int(items[1])
        nodes_set.add(node)
    nnodes = len(nodes_set)
    adj_matrix = sp.lil_matrix((nnodes, nnodes))
    node_attr_matrix = sp.lil_matrix((nnodes, nnodes))
    node_attr_matrix_acce = sp.lil_matrix((nnodes, num_node_attr))
    edge_attr_matrix = sp.lil_matrix((nnodes, nnodes)) 
    node_attr_file_open.close()

    # second going through to set up the adjacency (and possibly edge attribute) matrix
    graph_file_open = open(graph_file, 'r')
    for line in graph_file_open.readlines():
        items = line.strip().split(',')
        node_0 = int(items[0])
        node_1 = int(items[1])
        adj_matrix[node_0, node_1] = 1
        adj_matrix[node_1, node_0] = 1
        if (HAVE_EDGE_ATTR != 0):
            edge_attr_matrix[node_0, node_1] = int(items[2]) 
            edge_attr_matrix[node_1, node_0] = int(items[2])
    graph_file_open.close()

    # set up the node attribute matrix using a seperated file
    if (HAVE_NODE_ATTR != 0):
        node_attr_file_open = open(node_attr_file_name, 'r')
        for line in node_attr_file_open.readlines():
            items = line.strip().split(',')
            node = int(items[0])
            attr_of_node = int(items[1])
            node_attr_matrix_acce[node, attr_of_node] = 1
            node_attr_matrix[node, node] = attr_of_node

    return nnodes, adj_matrix, node_attr_matrix, edge_attr_matrix, node_attr_matrix_acce


if __name__ == '__main__':
    num_node_attr, num_edge_attr, node_attr_list, edge_attr_list = get_graph_info('datasets/' + DATASET_NAME + '-temporal/edge.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr.txt')
    nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, node_attr_matrix_acce_1 = build_matrices_accelerated(
        'datasets/' + DATASET_NAME + '-temporal/edge_sorted.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr.txt', max(node_attr_list) + 1
        )
    nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, node_attr_matrix_acce_2 = build_matrices_accelerated(
        'datasets/' + DATASET_NAME + '-temporal/edge_sorted_sub.txt', 'datasets/' + DATASET_NAME + '-temporal/node_attr_sub.txt', max(node_attr_list) + 1
        )

    initial_graph_file = open('datasets/' + DATASET_NAME + '-temporal/edge_sorted_initial.txt', 'r')
    start_line = len(initial_graph_file.readlines())
    initial_graph_file.close()
    graph_file = open('datasets/' + DATASET_NAME + '-temporal/edge_sorted.txt', 'r')
    all_line = graph_file.readlines()
    graph_file.close()
    end_line = len(all_line)


    # pre-knowledge
    h = construct.construct_pre_knowledge(nnodes_1, nnodes_2)
    # similarity
    v, time_consumption = solve.get_similarity(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, 
    nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, 
        node_attr_matrix_acce_1, node_attr_matrix_acce_2, h, 0.82)


    # get the ground truth match
    ground_truth_file = open('datasets/' + DATASET_NAME + '/ground_truth.txt', 'r')
    ground_truth = {}
    for line in ground_truth_file.readlines():
        items = line.strip().split(',')
        node_in_whole = int(items[0])
        node_in_sub = int(items[1])
        ground_truth[node_in_sub] = node_in_whole

    # calculate the hit rate using greedy match
    exp_match = solve.greedy_match(v, nnodes_1, nnodes_2)

    hit_rate1 = solve.check_greedy_hit1(exp_match, ground_truth, nnodes_2)



    t = 0
    counter = 0
    acc_record = []
    acc_record.append(hit_rate1)
    time_record = []
    time_record.append(time_consumption)
    for line_index in range(start_line, end_line):
        sub_change = 0
        items = all_line[line_index].strip().split(',')
        node1 = int(items[0])
        node2 = int(items[1])
        edge_attr = int(items[2])
        t += 1
        adj_matrix_1[node1, node2] = 0
        adj_matrix_1[node2, node1] = 0
        edge_attr_matrix_1[node1, node2] = 0
        edge_attr_matrix_1[node2, node1] = 0

        if is_in_subgraph(node1) and is_in_subgraph(node2):
            if edge_attr_matrix_2[match_from_whole_to_sub(node1), match_from_whole_to_sub(node2)] != 0:
                sub_change = 1
        if (sub_change == 0):
            continue
        # if comes to here, sub graph is changed
        counter += 1
        node_sub_1 = match_from_whole_to_sub(node1)
        node_sub_2 = match_from_whole_to_sub(node2)
        adj_matrix_2[node_sub_1, node_sub_2] = 0
        adj_matrix_2[node_sub_2, node_sub_1] = 0
        edge_attr_matrix_2[node_sub_1, node_sub_2] = 0
        edge_attr_matrix_2[node_sub_2, node_sub_1] = 0

        if (counter % GAP > 0):
            continue

        print("time:", t)
        print("line in all edges:", line_index + 1)
        print("subchange:", counter)

        # pre-knowledge
        h = construct.construct_pre_knowledge(nnodes_1, nnodes_2)
        # similarity
        v, time_consumption = solve.get_similarity(node_attr_list, edge_attr_list, nnodes_1, adj_matrix_1, node_attr_matrix_1, edge_attr_matrix_1, 
        nnodes_2, adj_matrix_2, node_attr_matrix_2, edge_attr_matrix_2, 
            node_attr_matrix_acce_1, node_attr_matrix_acce_2, h, 0.82)


        # calculate the hit rate using greedy match
        exp_match = solve.greedy_match(v, nnodes_1, nnodes_2)

        hit_rate1 = solve.check_greedy_hit1(exp_match, ground_truth, nnodes_2)
        acc_record.append(hit_rate1)
        time_record.append(time_consumption)
    pdb.set_trace()
