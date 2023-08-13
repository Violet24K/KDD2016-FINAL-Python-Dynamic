import numpy as np
DATASET_NAME = 'wikilens'
if DATASET_NAME == 'movielens-1m':
    from dataprocessing_temporal_movielens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'bitcoinalpha':
    from dataprocessing_temporal_bitcoinalpha import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub
elif DATASET_NAME == 'wikilens':
    from dataprocessing_temporal_wikilens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub


edge_dict = {}
edge_attr_set = set()
graph_file = open('./datasets/' + DATASET_NAME + '-temporal/edge_sorted.txt', 'r')
for line in graph_file.readlines():
    items = line.strip().split(',')
    node_in_whole_1 = items[0]
    node_in_whole_2 = items[1]
    attr = items[2]
    edge_attr_set.add(attr)
    edge_dict[(node_in_whole_1, node_in_whole_2)] = attr
graph_file.close()

pick = []

edge_attr_list = list(edge_attr_set)
keys_of = edge_dict.keys()
list_of = list(edge_dict.keys())
edge_attr_change_file = open('./datasets/' + DATASET_NAME + '-temporal/edge_attr_change.txt', 'w' )
counter = 0
for i in range(500):
    pick_index = np.random.randint(len(edge_dict.keys()))
    node1 = int(list_of[pick_index][0])
    node2 = int(list_of[pick_index][1])
    pre_attr = int(edge_dict[list_of[pick_index]])
    random_attr = pre_attr
    while (random_attr == pre_attr):
        ran = np.random.randint(len(edge_attr_list))
        random_attr = edge_attr_list[ran]
    edge_attr_change_file.write(str(node1) + ',' + str(node2) + ',' + str(random_attr) + '\n')
    if is_in_subgraph(node1) and is_in_subgraph(node2):
        counter += 1
print('num edges in sub:', counter)
    