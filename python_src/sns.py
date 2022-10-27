from synthesis_control.yosys_synthesis_did import yosys_synthesis_json_deploy
import orjson as json
from ff2ff_greedy_sample import *
import os
import time
from tensorflow import keras
import numpy as np
from timing_dataset import FF2FF_Data_Module
from models.circuitformer import Circuitformer
from texttable import Texttable
import torch
# torch.set_num_threads(1)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('verilog_design_filename')
parser.add_argument('-t', '--topname', help="Top Module Name of Design", default="")
parser.add_argument('-c', '--clockname', help="Name of Clock Signal in the Design", default="")
parser.add_argument('-p', '--powergating', help="JSON file for power gating coefficients", default="")
parser.add_argument('-m', '--models', help="The direcotry for all *.pt models", default="saved_models/")
parser.add_argument('-g', '--gpus', help="Number of GPUs to use.", default="")
parser.add_argument('-b', '--backend', help="Backend Only", action='store_true')
parser.add_argument('-r', '--reportpath', help="Output Path for SNS Report", default="./")
parser.add_argument('--tmp', help="Directory for storing Temporary files", default=".tmp/")
args = parser.parse_args()

BATCH_SIZE = 64


def design_info_to_vec(design_info):
    embed_dict_raw = open("{}/sns_embed_dict.json".format(args.models), 'rb').read()
    embed_dict = json.loads(embed_dict_raw)
    w2i = embed_dict['w2i']
    w2i['dff'] = len(embed_dict['i2w'])
    w2i['wire'] = len(embed_dict['i2w']) + 1
    w2i['nodes'] = len(embed_dict['i2w']) + 2
    w2i['edges'] = len(embed_dict['i2w']) + 3
    vec = np.zeros(len(embed_dict['i2w']) + 4, dtype=float)
    for k in design_info.keys():
        vec[w2i[k]] = design_info[k]
    return vec


def log10_mask_zero(array):
    return np.log10(array, out=np.zeros_like(array), where=(array != 0))


def log_norm(array, scale):
    array = np.log(array, out=np.zeros_like(array), where=(array != 0))
    array = array / scale
    return array


def log_denorm(enc, scale):
    enc = enc * scale
    return np.exp(enc)


g_props_scaling = 16.0
real_area_scaling = 16.0
pre_area_scaling = 16.0

pre_power_scaling = 16.0
real_power_scaling = 16.0


def infer_paths_timing(paths, checkpoint, hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 128,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 100.0
}):
    new_model = Circuitformer.load_from_checkpoint(checkpoint_path=checkpoint)
    new_model.use_gpu = -1
    new_model.hparams.batch_size = BATCH_SIZE
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", scaling='timing')
    new_model.datamodule = dm

    design = {}
    design['paths'] = paths
    timing, max_path = new_model.infer_on_circuit_timing(design, path_count=8192)
    total_num_paths = design['paths']

    # print(timing, max_path)
    return timing


def infer_paths_power(paths, checkpoint, hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 128,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 100.0
}):
    new_model = Circuitformer.load_from_checkpoint(checkpoint_path=checkpoint)
    new_model.use_gpu = -1
    new_model.hparams.batch_size = BATCH_SIZE
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", scaling='power')
    new_model.datamodule = dm

    design = {}
    design['paths'] = paths
    timing, max_path = new_model.infer_on_circuit_power(design, path_count=8192)
    total_num_paths = design['paths']

    # print(timing, max_path)
    return timing


if __name__ == "__main__":
    basename = os.path.basename(args.verilog_design_filename)
    seq_list = []
    g_properties = []
    profiling_dict = []
    rept = {}
    if args.backend:
        with open("{}/{}.paths".format(args.tmp, basename), "r") as f:
            seq_list = json.loads(f.read())
        with open("{}/{}.prop".format(args.tmp, basename), "r") as f:
            g_properties = json.loads(f.read())
        with open("{}/{}.prof".format(args.tmp, basename), "r") as f:
            profiling_dict = json.loads(f.read())
    else:
        profiling_dict = {}

        start = time.time()
        rtl_json = yosys_synthesis_json_deploy(args.verilog_design_filename,
                                               args.topname, rel_path=args.tmp, basename=basename)

        rtl = json.loads(rtl_json)
        end = time.time()
        yosys_time = end - start
        profiling_dict['yosys'] = yosys_time

        start = time.time()
        g = graph_from_json(rtl, args.topname)
        g_info = count_graph_cells(g)

        g_properties = {}
        g_properties['nodes'] = g.number_of_nodes()
        g_properties['edges'] = g.number_of_edges()
        g_properties.update(g_info)

        paths = greedy_paths_sample(g, k=100)
        print(len(paths), "paths generated")
        exit()
        seq_list = paths_to_sequence(paths)
        end = time.time()

        gir_time = end - start
        profiling_dict['gir'] = gir_time
        # print(len(seq_list))

        # timing = infer_timing(seq_list, checkpoint=args.models+"mobilebert_timing_final.pt")
        # print(timing)
        json_out = json.dumps(seq_list)
        json_file = open("{}/{}.paths".format(args.tmp, basename), "w")
        json_file.write(json_out)
        json_file.close()

        json_out = json.dumps(g_properties)
        json_file = open("{}/{}.prop".format(args.tmp, basename), "w")
        json_file.write(json_out)
        json_file.close()

        json_out = json.dumps(profiling_dict)
        json_file = open("{}/{}.prof".format(args.tmp, basename), "w")
        json_file.write(json_out)
        json_file.close()

    # Backend Part
    basename = basename.replace(".v", "")
    # print(basename)

    timing_model_path = os.path.join(args.models, "mobilebert_timing_selected.pt")
    power_model_path = os.path.join(args.models, "mobilebert_power_selected.pt")

    # Front-End
    start = time.time()
    timing_raw = infer_paths_timing(seq_list, checkpoint=timing_model_path)
    power_raw = infer_paths_power(seq_list, checkpoint=timing_model_path)
    rept['timing_raw'] = timing_raw
    rept['power_raw'] = power_raw
    end = time.time()
    bert_time = end - start
    profiling_dict['rtl-bert'] = bert_time

    power_raw *= 0.5

    # Backend
    g_props_vec_raw = design_info_to_vec(g_properties)
    g_props_vec = log_norm(g_props_vec_raw, g_props_scaling)
    g_props_vec_res = np.array([g_props_vec])

    pre_power_enc = np.array([[power_raw]])
    pre_power_enc = log_norm(pre_power_enc, 16.0)
    # pre_power_enc = log_norm(pre_power_enc, pre_power_scaling)
    # print(pre_power_enc.shape)
    # print(g_props_vec_res.shape)
    power_input_enc = np.concatenate([g_props_vec_res, pre_power_enc], axis=1)

    # print(power_input_enc.shape)

    # power_backend_input = np.concatenate([g_props_vec, pre_power_enc.reshape(-1, 1)], axis=1)

    # Model & Inference
    timing_mlp_path = os.path.join(args.models, "timing_mlp.h5")
    power_mlp_path = os.path.join(args.models, "power_mlp.h5")
    area_mlp_path = os.path.join(args.models, "area_mlp.h5")

    timing_model = keras.Sequential()
    timing_model.add(keras.layers.Dense(54, input_dim=1, activation='relu'))
    timing_model.add(keras.layers.Dense(1, activation='sigmoid'))
    power_model = keras.Sequential()
    power_model.add(keras.layers.Dense(54, input_dim=1, activation='relu'))
    power_model.add(keras.layers.Dense(1, activation='sigmoid'))
    area_model = keras.Sequential()
    area_model.add(keras.layers.Dense(54, input_dim=1, activation='relu'))
    area_model.add(keras.layers.Dense(1, activation='sigmoid'))

    start = time.time()
    timing_final = timing_model(np.array([g_props_vec])) * timing_raw
    rept['timing_final'] = timing_final.numpy()[0][0]

    power_final = power_model(power_input_enc)
    power_final = log_denorm(power_final, 16.0)
    rept['power_final'] = power_final[0][0]

    area_final = area_model(np.array([g_props_vec]))
    area_final = log_denorm(area_final, pre_area_scaling)
    rept['area_final'] = area_final[0][0]

    end = time.time()
    mlp_time = end - start
    profiling_dict['aggr-mlp'] = mlp_time

    t = Texttable()
    print("\n\n\n")
    print("SNS Report: ")
    t.add_rows(
        [['Module', "Power (mW)", "Timing (ps)", "Area (um^2)"], [args.topname, power_final, timing_raw, area_final]])
    print(t.draw())

    # Output Report
    for k in rept:
        rept[k] = float(rept[k])

    rept['prifoling'] = profiling_dict
    json_out = json.dumps(rept)
    report_path = "{}/{}.rept".format(args.reportpath, basename)
    print("Reports Writes to: ", report_path)
    json_file = open(report_path, "w")
    json_file.write(json_out)
    json_file.close()
