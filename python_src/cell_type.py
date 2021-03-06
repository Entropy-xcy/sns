cell_type_list = {
    "add": [8, 16, 32, 64],
    "sub": [8, 16, 32, 64],
    "mul": [8, 16, 32, 64],
    "mux": [0, 0],
    "not": [0, 0],
    "and": [0, 0], 
    "or": [0, 0],
    "xor": [0, 0],
    "shl": [4, 8, 16, 32, 64],
    "sh": [4, 8, 16, 32, 64],
    "shr": [4, 8, 16, 32, 64],
    "reduce_and": [4, 8, 16, 32, 64],
    "reduce_or": [4, 8, 16, 32, 64],
    "reduce_xor": [4, 8, 16, 32, 64],
    "eq": [8, 16, 32, 64],
    "ne": [8, 16, 32, 64],
    "le": [8, 16, 32, 64],
    "lt": [8, 16, 32, 64],
    "ge": [8, 16, 32, 64],
    "gt": [8, 16, 32, 64],
    "lgt": [8, 16, 32, 64],
    "div": [8, 16, 32, 64],
    "mod": [8, 16, 32, 64]
}

grouping_dict = {
    "mux": "mux",
    "not": "not",
    "logical_not": "not",
    "logic_not": "not",
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
