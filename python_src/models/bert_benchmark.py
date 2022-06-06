from circuitformer import Circuitformer
import torch as th
import time
from math import ceil

POWER_CHK_POINT = "saved_models/mobilebert_power_final.pt"
TIMING_CHK_POINT = "saved_models/mobilebert_timing_final.pt"
compiled_timing = 'ff2ff_timing_bert.pt'
compiled_power = 'ff2ff_power_bert.pt'
compiled_model = compiled_timing
batchsize = 128
num_paths = 51200

def save():
    model = Circuitformer.load_from_checkpoint(checkpoint_path=TIMING_CHK_POINT)

    dummy_inputs = th.ones(batchsize, 512, dtype=int)
    model = model.to("cuda:0")
    dummy_inputs = dummy_inputs.to("cuda:0")

    traced_cell = th.jit.trace(model, (dummy_inputs))
    traced_cell.save('ff2ff_timing_bert_gpu.pt')

def gpu_bench():
    model = th.jit.load('ff2ff_timing_bert_gpu.pt')
    dummy_inputs = th.ones(batchsize, 512, dtype=int, device="cuda:0")

    start = time.time()
    for i in range(ceil(num_paths / batchsize)):
        out = model(dummy_inputs)
    end = time.time()

def another_main():
    model = th.jit.load(compiled_model)
    dummy_inputs = th.ones(batchsize, 512, dtype=int)
    
    start = time.time()
    for i in range(ceil(num_paths / batchsize)):
        out = model(dummy_inputs)
    end = time.time()

    print("Time Elapsed: {}", end-start)

def gpu_benchmark():
    model = Circuitformer.load_from_checkpoint(checkpoint_path=POWER_CHK_POINT)
    model.zero_grad()

    dummy_inputs = th.ones(batchsize, 512, dtype=int)

    model = model.to("cuda:0")
    dummy_inputs = dummy_inputs.to("cuda:0")

    start = time.time()
    for i in range(ceil(num_paths / batchsize)):
        out = model(dummy_inputs)
    end = time.time()

    print("Time Elapsed: {}", end-start)


if __name__ == "__main__":
    # gpu_benchmark()
    save()
    gpu_bench()

    
