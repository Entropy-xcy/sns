from synpred import design, ir, asic, fs
import os


grouping_dict = {
    "mux": "mux",
    "not": "not",
    "logical_not": "not",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "shl": "sh",
    "shr": "sh",
    "sshl": "sh",
    "sshr": "sh",
    "reduce_and": "reduce_and",
    "reduce_or": "reduce_or",
    "reduce_xor": "reduce_xor",
    "eq": "eq",
    "ne": "eq",
    "lt": "lgt",
    "le": "lgt",
    "ge": "lgt",
    "gt": "lgt",
    "add": "add",
    "sub": "add",
    "mul": "mul",
    "div": "div",
    "mod": "mod"
}

width_independent_alphabet = ["mux", "not", "and", "or", "xor"]
width_dependent_alphabet = ["sh4", "sh8", "sh16", "sh32", "sh64",
                           "reduce_and4", "reduce_and8", "reduce_and16", "reduce_and32", "reduce_and64",
                           "reduce_or4", "reduce_or8", "reduce_or16", "reduce_or32", "reduce_or64", 
                           "reduce_xor4", "reduce_xor8", "reduce_xor16", "reduce_xor32", "reduce_xor64",
                           "eq8", "eq16", "eq32", "eq64",
                           "lgt8", "lgt16", "lgt32", "lgt64",
                           "add8", "add16", "add32", "add64",
                           "mul8", "mul16", "mul32", "mul64",
                           "div8", "div16", "div32", "div64",
                           "mod8", "mod16", "mod32", "mod64"]

alphabet = width_independent_alphabet + width_dependent_alphabet

def rawseq2encoding(seq):
    out_encoding = []
    for s in seq:
        row = s.split(",")
        cell_type = row[0]
        cell_encoding = grouping_dict[cell_type]
        if cell_encoding in width_independent_alphabet:
            assert cell_encoding in alphabet
            out_encoding.append(cell_encoding)
        else:
            assert cell_encoding+row[1] in alphabet
            out_encoding.append(cell_encoding+row[1])
    return out_encoding

QUERY = {"description": "ff2ff path dataset"}
ORDER = [("did", 1)]

ff2ff_designs = design.find(QUERY).sort(ORDER)
ff2ff_counts = design.count_documents(QUERY)

print("Raw dataset length: ", ff2ff_counts)


seq_list = []
power_list = []

for i in range(ff2ff_counts):
    did = ff2ff_designs[i]["did"]
    DID_QUERY = {"did": did}
    seq = ir.find_one(DID_QUERY)['result']
    power = asic.find_one(DID_QUERY)['power']
    timing = asic.find_one(DID_QUERY)['timing']
    status = asic.find_one(DID_QUERY)['status']
    if status == "done" and timing > 200.0 and len(seq) >= 0 and power > 0:
    # if status == "done":
        seq_list.append(seq)
        # assert timing > 200.0
        power_list.append(power)
    else:
        print("Not Available: {}".format(did))
    
    if i%25 == 0:
        print("{}%".format(i * 100.0 / ff2ff_counts))

seq_encoded_list = []
for s in seq_list:
    seq_encoded_list.append(rawseq2encoding(s))

print("Alphabet Length: {}".format(len(alphabet)))
print("Dataset Samples: {}".format(len(seq_encoded_list)))

import json

dataset = (alphabet, seq_encoded_list, power_list)
j = json.dumps(dataset, separators=(',', ':'))
jsonoutfile = open("ff2ff_dataset_power.json", "w+")
jsonoutfile.write(j)
jsonoutfile.close()

