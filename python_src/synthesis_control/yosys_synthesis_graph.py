from subprocess import Popen, PIPE, STDOUT
import pandas as pd
from synpred import download_design, upload_result, init_job, fs
import os
import sys
import time
import networkx as nx
from dgl import DGLGraph
from dgl.data.utils import save_graphs
import dgl
from table2graph import table2_ud_nf_graph
from yosys_synthesis_did import yosys_synthesis_json_did
from ff2ff_timing_inferflow import graph_from_json

DGL_PIPE = "_dg.pipe"

def rm_dict_dollar_key(dict):
    for k in dict.copy().keys():
        new_key = k.replace("$", "")
        dict[new_key] = dict[k]
        del dict[k]
    return dict

def yosys_synthesis_table(did, filename, top_name, rel_path="yosys/"):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    tcl = 'read_verilog {}/{}; proc; flatten; write_table {}/{}.table;'.format(rel_path, filename, rel_path, filename).encode()
    yosys_out = p.communicate(input=tcl)[0]
    print("Yosys Finished!")

    table_filename = "{}/{}.table".format(rel_path, filename)
    gexf_filename = "{}/{}.gexf".format(rel_path, filename)
    syn_table = open(table_filename, "r")
    table_data = pd.read_csv(table_filename, sep = '\t', header = None)

    TOP_NAME = top_name

    # Process Top Module Only
    rtl_tb = table_data
    top_tb = rtl_tb[rtl_tb[0] == TOP_NAME]
    G = nx.Graph()

    for index, row in top_tb.iterrows():
        mod_name = row[0]
        cell_name = row[1]
        cell_type = row[2]
        cell_port = row[3]
        sig_dir = row[4]
        sig_name = row[5]
        G.add_edge(cell_name, sig_name)
    
    dg = dgl.from_networkx(G)

    print("Graph Info:")
    print(G.number_of_edges())
    print(G.number_of_nodes())
    DGL_PIPE_PATH = str(did) + DGL_PIPE

    save_graphs(DGL_PIPE_PATH, dg)
    dg_bin = open(DGL_PIPE_PATH, "rb").read()
    os.remove(DGL_PIPE_PATH)

    return dg_bin

def yosys_synthesis_table_only(did, filename, top_name, rel_path="yosys/"):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    tcl = 'read_verilog {}/{}; proc; flatten; write_table {}/{}.table;'.format(rel_path, filename, rel_path, filename).encode()
    yosys_out = p.communicate(input=tcl)[0]
    print("Yosys Finished!")

    table_filename = "{}/{}.table".format(rel_path, filename)
    gexf_filename = "{}/{}.gexf".format(rel_path, filename)
    syn_table = open(table_filename, "r")
    table_data = pd.read_csv(table_filename, sep = '\t', header = None)
    return table_data

def yosys_synthesis_did(did, stype, rel_path="yosys/"):
    if stype == "raw_graph":
        # Init Job
        init_job(did, "ir", stype)
        
        # Prepare for Synthesis
        des = download_design(did)
        job_name = "{}_{}_{}".format(str(did), "ir", stype)
        top_name = des["top_name"]
        v_filepath = os.path.join(rel_path, job_name)
        v_filename = des['top_name']+ ".v"
        # Create Directory if not exist
        try:
            os.mkdir(v_filepath)
        except:
            print("Path Already Exist.")
        v_file = open(v_filepath + "/" + v_filename, "wb")
        v_file.write(des['source'])
        v_file.close()
        print("Did={}, {} Downloaded".format(str(did), v_filename))

        start = time.time()

        # Start Synthesis
        dg_bin = yosys_synthesis_table(did, v_filename, top_name, rel_path=v_filepath)

        end = time.time()
        time_elapsed = end - start

        # Upload Result
        fid = fs.put(dg_bin, filename="{}.dg".format(did))
        results = {"result": fid, "runtime": time_elapsed}
        upload_result(did, "ir", stype, results)
        print("Result Uploaded!")
    elif stype == "ud_nf_graph":
        # Init Job
        init_job(did, "ir", stype)
        
        # Prepare for Synthesis
        des = download_design(did)
        job_name = "{}_{}_{}".format(str(did), "ir", stype)
        top_name = des["top_name"]
        v_filepath = os.path.join(rel_path, job_name)
        v_filename = des['top_name']+ ".v"
        # Create Directory if not exist
        try:
            os.mkdir(v_filepath)
        except:
            print("Path Already Exist.")
        v_file = open(v_filepath + "/" + v_filename, "wb")
        v_file.write(des['source'])
        v_file.close()
        print("Did={}, {} Downloaded".format(str(did), v_filename))

        start = time.time()

        # Synthesis For table
        print(v_filepath)
        table_data = yosys_synthesis_table_only(did, v_filename, top_name, rel_path=v_filepath)

        # Start Synthesis
        dg_bin, num_nodes, num_edges = table2_ud_nf_graph(did, table_data, top_name, rel_path=v_filepath)

        end = time.time()
        time_elapsed = end - start

        # Upload Result
        fid = fs.put(dg_bin, filename="{}.dg".format(did))
        results = {"result": fid, "runtime": time_elapsed, "nodes": num_nodes, "edges": num_edges}
        upload_result(did, "ir", stype, results)
        print("Result Uploaded!")
    elif stype == "gexf":
        rtl, top_name = yosys_synthesis_json_did(did)
        g = graph_from_json(rtl, top_name)
        g_filepath = "./graph_out/{}.gexf".format(did)
        nx.write_gexf(g, g_filepath)
    else:
        raise ValueError("stype of {} is nor supported".format(stype))

if __name__ == "__main__":
    did = int(sys.argv[1])
    yosys_synthesis_did(did, "gexf")
