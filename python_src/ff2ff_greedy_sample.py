from math import ceil
from multiprocessing import Pool
import numpy as np
from ff2ff_timing_inferflow import *
import networkx as nx
import sys
from cell_type import *

sys.setrecursionlimit(5000)

"""
def sample_path_recur(g, curr_node):
    curr_name = g.nodes[curr_node]['cell_type']
    curr_width = g.nodes[curr_node]['cell_width']
    if curr_name == "dff":
        return [], curr_node
    else:
        successors = list(g.successors(curr_node))
        if len(successors) == 0:
            return [], None
        succ = random.choice(successors)
        succ_path, end_dff = sample_path_recur(g, succ)
        return [(curr_name, int(curr_width))] + succ_path, end_dff

def sample_path_node(g, start_node):
    successors = list(g.successors(start_node))
    chosen_succ = random.choice(successors)
    return sample_path_recur(g, chosen_succ)
"""


def sample_path_node(g, start_node):
    path = []
    op_path = []
    path.append((g.nodes[start_node]['cell_type'], g.nodes[start_node]['cell_width']))
    current_node = start_node
    end = False
    end_dff = None
    while not end:
        successors = list(g.successors(current_node))
        if len(successors) == 0:
            break
        sucessee = random.choice(successors)
        path.append((g.nodes[sucessee]['cell_type'], g.nodes[sucessee]['cell_width']))
        current_node = sucessee
        if g.nodes[sucessee]['cell_type'] == "dff":
            end = True
            end_dff = sucessee
    return path, end_dff


def sample_paths_node(g, start_node, num):
    ret_paths = []
    end_dffs = set()
    for _ in range(num):
        path, edff = sample_path_node(g, start_node)
        ret_paths.append(path)
        end_dffs.add(edff)
    return ret_paths, end_dffs


def greedy_paths_from_node(g, start_node, k=10):
    path_sampled = 0
    target_num_samples = k * 2
    ret_paths = []
    end_dffs = set()
    # print(target_num_samples)
    while path_sampled < target_num_samples:
        num_to_sample = target_num_samples - path_sampled
        paths, edff = sample_paths_node(g, start_node, num_to_sample)
        ret_paths += paths
        end_dffs = end_dffs.union(edff)
        # print(end_dffs)
        path_sampled += num_to_sample
        target_num_samples = k * len(end_dffs)
    return ret_paths, end_dffs, path_sampled


def greedy_paths_sample(g, power_gating_dict={}, k=10):
    dff_nodes = []
    ret_paths = []
    for node_id in search_nodes(g, {"==": [("cell_type",), "dff"]}):
        dff_nodes.append(node_id)
    dff_nodes_pg = np.ones(len(dff_nodes))

    for i in range(len(dff_nodes)):
        d = dff_nodes[i]
        for key in power_gating_dict.keys():
            if key in d:
                dff_nodes_pg[i] = power_gating_dict[key]

    for i in range(len(dff_nodes)):
        d = dff_nodes[i]
        pg = dff_nodes_pg[i]
        paths, end_dffs, paths_sampled = greedy_paths_from_node(g, d, k=k)
        # print(len(paths))
        if pg != 1.0:
            # print("Power Gated!")
            paths = random.choices(paths, k=ceil(paths_sampled * pg))
        ret_paths += paths
    return ret_paths


def count_graph_cells(g):
    count_dict = {}
    for a in alphabet:
        count_dict[a] = 0
    count_dict['wire'] = 0
    count_dict['dff'] = 0

    for n in g.nodes():
        cell = (g.nodes[n]['cell_type'], g.nodes[n]['cell_width'])
        if g.nodes[n]['cell_type'] == 'wire':
            count_dict['wire'] += 1
        elif 'dff' in g.nodes[n]['cell_type']:
            count_dict['dff'] += g.nodes[n]['cell_width']
        elif g.nodes[n]['cell_type'] in grouping_dict.keys():
            seq = sequence_op_from_cell(cell)
            # print(cell)
            count_dict[seq] += 1

    return count_dict


if __name__ == "__main__":
    did = int(sys.argv[1])
    rtl, top_name = yosys_synthesis_json_did(did)

    g = graph_from_json(rtl, top_name)
    g_info = count_graph_cells(g)

    g_properties = {}
    g_properties['nodes'] = g.number_of_nodes()
    g_properties['edges'] = g.number_of_edges()
    g_properties.update(g_info)

    paths = greedy_paths_sample(g, k=5)
    seq_list = paths_to_sequence(paths)

    json_out = json.dumps(seq_list)
    json_file = open("path_out/{}.paths".format(did), "w")
    json_file.write(json_out)
    json_file.close()

    json_out = json.dumps(g_properties)
    json_file = open("path_out/{}.prop".format(did), "w")
    json_file.write(json_out)
    json_file.close()
