from synpred_db import *
from subprocess import Popen, PIPE, STDOUT

# Infer the Name of the Top Module and Clock 
def infer_top_clk(filename):
    return infer_top_name(filename), infer_clock_name(filename)

# Infer the Name of the Top Module of the design source code
def infer_top_name(filename):
    p = Popen(['yosys'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)    
    grep_stdout = p.communicate(input='read_verilog {}; hierarchy -auto-top;'.format(filename).encode())[0]
    out = grep_stdout.decode()
    top_name = out[out.find("Top module:  \\"):]
    top_name = top_name[top_name.find("\\")+1: top_name.find("\n")]
    return top_name

# Infer the Name of the Clock of the design source code
def infer_clock_name(filename):
    src = open(filename, "r").read()
    clock_inside = src.find(" clock") >= 0
    clk_inside = src.find(" clk") >= 0
    if clock_inside and not clk_inside:
        return "clock"
    elif clk_inside and not clock_inside:
        return "clk"
    else:
        return input("Auto Clock Inference Failed, Please Enter the name of the Clock:")

if __name__ == "__main__":
    print(infer_clock_name("Xmcmc.v"))
    print(infer_top_name("Xmcmc.v"))
