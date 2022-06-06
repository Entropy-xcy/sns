from subprocess import Popen, PIPE, STDOUT
import pandas as pd
from synpred import download_design, upload_result, init_job, asic
import os
import sys
import time
import json


def rm_dict_dollar_key(dict):
    for k in dict.copy().keys():
        if "$" in k:
            new_key = k.replace("$", "")
            dict[new_key] = dict[k]
            del dict[k]
    return dict

def yosys_synthesis_count(filename, top_name, rel_path="yosys/"):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    tcl = 'read_verilog {}/{}; hierarchy -top {}; proc; flatten; memory_dff; memory_share; memory_collect; memory_map; clean; write_table {}/{}.table;'.format(rel_path, filename, top_name, rel_path, filename).encode()
    yosys_out = p.communicate(input=tcl)[0]
    print("Yosys Finished!")

    table_filename = "{}/{}.table".format(rel_path, filename)
    syn_table = open(table_filename, "r")
    table_data = pd.read_csv(table_filename, sep = '\t', header = None)
    count_list = {}
    keys = table_data[2].unique()
    
    count_table = table_data[table_data[0] == top_name][2].value_counts()
    count_dict = count_table.to_dict()
    print("Counting Finished!")
    count_dict = rm_dict_dollar_key(count_dict)
    return count_dict

def yosys_synthesis_json(filename, top_name, rel_path="yosys/"):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    tcl = 'read_verilog {}/{}; hierarchy -top {}; proc; flatten; memory_dff; memory_share; memory_collect; memory_map; clean; write_json {}/{}.json;'.format(rel_path, filename, top_name, rel_path, filename).encode()
    yosys_out = p.communicate(input=tcl)[0]
    print("Yosys Finished!")

    json_filename = "{}/{}.json".format(rel_path, filename)
    rtl_json = open(json_filename, "r").read()
    os.remove(json_filename)
    return rtl_json

def yosys_synthesis_json_deploy(filename, top_name, basename, rel_path="yosys/"):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    tcl = 'read_verilog {}; hierarchy -top {}; proc; flatten; memory_dff; memory_share; memory_collect; memory_map; clean; write_json {}/{}.json;'.format(filename, top_name, rel_path, basename).encode()
    yosys_out = p.communicate(input=tcl)[0]
    yosys_outfile = open("{}/{}.log".format(rel_path, basename), "wb")
    yosys_outfile.write(yosys_out)
    yosys_outfile.close()

    print("Yosys Finished!")

    json_filename = "{}/{}.json".format(rel_path, basename)
    rtl_json = open(json_filename, "r").read()
    os.remove(json_filename)
    return rtl_json

def yosys_synthesis_json_did(did, rel_path="yosys/"):
    # Prepare for Synthesis
    stype = 'json'
    des = download_design(did)
    job_name = "{}_{}_{}".format(str(did), "ir", stype)
    v_filepath = os.path.join(rel_path, job_name)
    v_filename = des['top_name']+ ".v"
    topname = des['top_name']

    # Create Directory if not exist
    try:
        os.mkdir(v_filepath)
    except:
        print("Path Already Exist.")
    v_file = open(v_filepath + "/" + v_filename, "wb")
    v_file.write(des['source'])
    v_file.close()
    print("Did={}, {} Downloaded".format(str(did), v_filename))

    # Start Synthesis
    rtl_json = yosys_synthesis_json(v_filename, topname, rel_path=v_filepath)

    # Process Json
    rtl = json.loads(rtl_json)
    # print(rtl.keys())
    return rtl, topname

def yosys_synthesis_did(did, stype, rel_path="yosys/"):
    assert stype == "count"

    # Init Job
    init_job(did, "ir", stype)

    # Prepare for Synthesis
    des = download_design(did)
    job_name = "{}_{}_{}".format(str(did), "ir", stype)
    v_filepath = os.path.join(rel_path, job_name)
    v_filename = des['top_name']+ ".v"
    topname = des['top_name']
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
    count_dict = yosys_synthesis_count(v_filename, topname, rel_path=v_filepath)
    print(count_dict)
    end = time.time()
    time_elapsed = end - start
    # Upload Results
    results = {"result": count_dict, "runtime": time_elapsed}
    upload_result(did, "ir", "count", results)
    print("Result Uploaded!")


if __name__ == "__main__":
    did = sys.argv[1]
    # yosys_synthesis_did(int(did), "count")
    yosys_synthesis_json_did(int(did))
