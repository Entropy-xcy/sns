from synthesis_control.yosys_synthesis_did import yosys_synthesis_json_did
import networkx as nx
from networkx_query import search_nodes
import random
from cell_type import *
import sys
import time
import json
import os

def process_cell_type(cell_json):
    cell_type_raw = cell_json['type']
    cell_type = cell_type_raw.replace("$", "")
    
    cell_param_raw = cell_json['parameters']
    max_width = 0
    for k in cell_param_raw.keys():
        if "WIDTH" in k:
            # print(cell_param_raw[k])
            # max_width = max(max_width, int(cell_param_raw[k], 2))
            # print(type(cell_param_raw[k]))
            max_width = max(max_width, int(cell_param_raw[k], 2))
    
    return cell_type, max_width

def graph_from_json(rtl, top_name):
    g = nx.DiGraph()
    cells = rtl['modules'][top_name]['cells']
    for k in cells.keys():
        cell_json = cells[k]

        cell_type, cell_width = process_cell_type(cell_json)
        g.add_node(k, cell_type=cell_type, cell_width=cell_width)

        cell_type_raw = cell_json['type']
        cell_type = cell_type_raw.replace("$", "")

        connections = cell_json['connections']
        if 'module_not_derived' in cell_json['attributes'].keys():
            print("Skip One Cell")
            continue
        if 'port_directions' not in cell_json.keys():
            print("Error!")
            print(cell_json)
        port_directions = cell_json['port_directions']

        for port in connections.keys():
            bits = connections[port]
            direction = port_directions[port]

            for b in bits:
                g.add_node(b, cell_type='wire', cell_width=1)
                if direction == 'input':
                    g.add_edge(b, k)
                else:
                    g.add_edge(k, b)
    return g

def sample_ff2ff_from_node(g, start_node):
    path = []
    op_path = []
    path.append((g.nodes[start_node]['cell_type'], g.nodes[start_node]['cell_width']))
    current_node = start_node
    end = False
    while not end:
        successors = list(g.successors(current_node))
        if len(successors) == 0:
            break
        sucessee = random.choice(successors)
        path.append((g.nodes[sucessee]['cell_type'], g.nodes[sucessee]['cell_width']))
        current_node = sucessee
        if g.nodes[sucessee]['cell_type'] == "dff":
            end = True
    return path

def sample_ff2ff_from_graph(g, num=1):
    dff_nodes = []
    for node_id in search_nodes(g, {"==": [("cell_type",), "dff"]}):
        dff_nodes.append(node_id)
    
    print("{} DFF Nodes".format(len(dff_nodes)))

    # selected_start = random.sample(dff_nodes, num)
    # random.choises(x, k=v)
    selected_start = random.choices(dff_nodes, k=num)
    ret = []
    for s in selected_start:
        ret.append(sample_ff2ff_from_node(g, s))
    
    return ret

def count_ff2ff_paths_from_node(g, start_node):
    dff_nodes = []
    for node_id in search_nodes(g, {"==": [("cell_type",), "dff"]}):
        dff_nodes.append(node_id)
    
    all_paths = []
    for dffn in dff_nodes:
        for path in nx.all_simple_paths(g, source=start_node, target=dffn):
            print(p)

def count_ff2ff_paths_from_node_test(did):
    rtl, top_name = yosys_synthesis_json_did(did)
    # print(rtl['modules'][top_name]['cells'])

    g = graph_from_json(rtl, top_name)

    dff_nodes = []
    for node_id in search_nodes(g, {"==": [("cell_type",), "dff"]}):
        dff_nodes.append(node_id)
    
    count_ff2ff_paths_from_node(g, dff_nodes[0])


def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 


def sequence_op_from_cell(cell):
    cell_type = cell[0]
    cell_width = cell[1]
    
    if cell_type not in grouping_dict.keys():
        return ""
    
    cell_type = grouping_dict[cell_type]
    
    cell_width = closest(cell_type_list[cell_type], cell_width)
    
    ret = ""
    if cell_width == 0:
        ret = str(cell_type)
    else:
        ret = str(cell_type)+str(cell_width)
    assert ret in alphabet 
    return ret


def paths_to_sequence(paths):
    ret = []
    for p in paths:
        sub_seq = []
        for cell in p:
            seq = sequence_op_from_cell(cell)
            if seq != "":
                sub_seq.append(seq)
        ret.append(sub_seq)
    return ret

from lr_model import predict_seq as predict_seq_lr

def infer_max_timing(seq_list):
    max_timing = 0
    max_seq = []
    for s in seq_list:
        pred = predict_seq_lr(s)[0]
        if pred > max_timing:
            max_timing = pred
            max_seq = s
    return max_timing, max_seq


def sample_ff2ff_from_rtl(did):
    rtl, top_name = yosys_synthesis_json_did(did)
    # print(rtl['modules'][top_name]['cells'])

    g = graph_from_json(rtl, top_name)
    g_properties = {}
    g_properties['nodes'] = g.number_of_nodes()
    g_properties['edges'] = g.number_of_edges()

    paths = sample_ff2ff_from_graph(g, num=1000)
    seq_list = paths_to_sequence(paths)
    return seq_list, g_properties


def timing_inference_json(json, top_name, samples=1000):
    # Turn json to Graph
    g = graph_from_json(json, top_name)
    g_properties = {}
    g_properties['nodes'] = g.number_of_nodes()
    g_properties['edges'] = g.number_of_edges()
    
    # Sample From Graph
    paths = sample_ff2ff_from_graph(g, num=samples)
    
    # Convert Path to sequences
    seq_list = paths_to_sequence(paths)
    
    # Infering Timing
    max_timing, max_seq = infer_max_timing(seq_list)
    
    return max_timing, max_seq, seq_list, g_properties

def timing_inference_did(did, samples=100):
    # Prepare Json
    rtl, top_name = yosys_synthesis_json_did(did)
    timing, max_seq, seq_list, g_properties= timing_inference_json(rtl, top_name, samples=samples)
    return timing, max_seq, seq_list, g_properties
    
if __name__ == "__main__":
    did = int(sys.argv[1])
    start_time = time.time()
    timing, max_seq, seq_list, g_properties = timing_inference_did(did, samples=100000)
    end_time = time.time()

    json_out = json.dumps(seq_list)
    json_file = open("path_out/{}.paths".format(did), "w")
    json_file.write(json_out)
    json_file.close()
    
    json_out = json.dumps(g_properties)
    json_file = open("path_out/{}.prop".format(did), "w")
    json_file.write(json_out)
    json_file.close()

    print("Max timing: {}".format(timing))
    print("Runtime: {}".format(end_time - start_time))
    print("Max Sequence: {}".format(max_seq))


if __name__ == "__main__":
    count_ff2ff_paths_from_node_test(1)
