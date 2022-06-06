import torch as th
import numpy as np
import orjson
from torch import random
from torch._C import dtype
from torch.nn import *
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import pytorch_lightning as pl

# from hyper_params import *
MAX_LEN = 512

class FF2FF_Dataset(Dataset):
    """
    def normalize_timing(self, timing):
        return (timing - self.timing_mean) / self.timing_std

    def denormalize_timing(self, out):
        return out * self.timing_std + self.timing_mean
    """
    def normalize_timing(self, timing):
        return timing / self.timing_scaling

    def denormalize_timing(self, out):
        return out * self.timing_scaling

    def normalize_counts(self, counts):
        return counts / self.wc_scaling

    def denormalize_counts(self, out):
        return out * self.wc_scaling

    def __init__(self, dataset_json_filename, scaling="timing", verbose=False):
        print("Loading JSON Dataset...")
        dataset_json = open(dataset_json_filename, "r").read()
        dataset = orjson.loads(dataset_json)
        alphabet = ['-'] + dataset[0]
        sequence = dataset[1]
        timing = dataset[2]
        assert len(sequence) == len(timing)
        # self.alphabet = alphabet

        self.timing = th.Tensor(timing)
        self.timing_std = th.std(self.timing)
        self.timing_mean = th.mean(self.timing)

        rmv_index = []
        new_timing = []
        new_sequence = []
        for i in range(len(timing)):
            if timing[i] < self.timing_mean - 2* self.timing_std or \
                timing[i] > self.timing_mean + 2* self.timing_std:
                rmv_index.append(i)
            else:
                new_timing.append(timing[i])
                new_sequence.append(sequence[i])
        if verbose:
            print("To Remove: ", rmv_index)

        timing = new_timing
        sequence = new_sequence
        self.sequence = sequence
        self.timing = th.Tensor(timing)

        self.vocab = {}
        self.vocab['i2w'] = alphabet
        self.vocab['w2i'] = {}
        for i in range(len(alphabet)):
            self.vocab['w2i'][alphabet[i]] = i
        self.wc_len = len(alphabet)
        self.wc_scaling = 4000.0

        if scaling == "timing":
            self.timing_scaling = 4000.0
        elif scaling == "power":
            self.timing_scaling = 20.0
        else:
            pass
        
        if verbose:
            print("Alphabet Length: {}".format(len(alphabet)))
            print("Alphabet: ", alphabet)
            print("Dataset Samples: {}".format(len(sequence)))
            print("Example Sequence: {}..".format(sequence[0][: 5]))
            print("Example Timing: {}".format(timing[0]))
    
    def __len__(self):
        assert len(self.timing) == len(self.sequence)
        return len(self.timing)

    def encode_seq(self, sequence):
        sequence_enc = th.LongTensor([self.vocab['w2i'][s] for s in sequence])
        unique_w, counts = th.unique(sequence_enc, return_counts=True)
        wc_enc = th.zeros([self.wc_len], dtype=th.float32)
        for i in range(len(unique_w)):
            wc_enc[unique_w[i]] = counts[i] 
        wc_enc = self.normalize_counts(wc_enc)
        sequence_enc_ret = th.zeros([MAX_LEN], dtype=th.long)

        sequence_len = len(sequence)
        if sequence_len > MAX_LEN:
            sequence_enc_ret = sequence_enc[0:MAX_LEN]
        else:
            sequence_enc_ret[0:sequence_len] = sequence_enc

        return sequence_enc_ret, wc_enc

    def __getitem__(self, idx):
        sequence = self.sequence[idx]
        timing_enc = self.normalize_timing(self.timing[idx])

        sequence_enc_ret, wc_enc = self.encode_seq(sequence)

        return sequence_enc_ret, wc_enc, timing_enc
    
    def encode_seq_old(self, sequence):
        sequence_enc = th.LongTensor([self.vocab['w2i'][s] for s in sequence])
        sequence_enc_ret = th.zeros([MAX_LEN], dtype=th.long)
        sequence_len = len(sequence)
        if sequence_len > MAX_LEN:
            sequence_enc_ret = sequence_enc[0:MAX_LEN]
        else:
            sequence_enc_ret[0:sequence_len] = sequence_enc
        
        return sequence_enc_ret


class FF2FF_Dataset_PreCompute(FF2FF_Dataset):
    def __init__(self, dataset_json_filename, scaling='timing'):
        super().__init__(dataset_json_filename, scaling=scaling)
        dataset_len = super().__len__()
        self.sequence_enc_precompute = th.zeros((dataset_len, MAX_LEN), dtype=th.long)
        self.wc_enc_precompute = th.zeros((dataset_len, self.wc_len), dtype=th.float32)
        self.timing_enc_precompute = th.zeros((dataset_len), dtype=th.float32)
        for i in range(dataset_len):
            sequence_enc_ret, wc_enc, timing_enc = super().__getitem__(i)
            self.sequence_enc_precompute[i] = sequence_enc_ret
            self.wc_enc_precompute[i] = wc_enc
            self.timing_enc_precompute[i] = timing_enc

    def __getitem__(self, idx):
        return self.sequence_enc_precompute[idx], \
               self.wc_enc_precompute[idx], \
               self.timing_enc_precompute[idx]


class FF2FF_Data_Module(pl.LightningDataModule):
    def __init__(self, dataset_path, val_set_percentage=30, num_workers=0, 
                    persistent_workers=False, pin_memory=True, scaling='timing'):
        super().__init__()
        # self.batch_size = batch_size
        self.num_workers = num_workers

        self.ff2ff_timing_dataset = FF2FF_Dataset_PreCompute(dataset_path, scaling=scaling)
        val_set_len = int(len(self.ff2ff_timing_dataset) * val_set_percentage / 100.0)
        self.ff2ff_timing_train, self.ff2ff_timing_test = random_split(self.ff2ff_timing_dataset, 
                                                    [len(self.ff2ff_timing_dataset) - val_set_len, val_set_len])
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
    
    def train_dataloader(self, batch_size):
        return DataLoader(self.ff2ff_timing_train, batch_size, shuffle=True, 
                                    pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self, batch_size):
        return DataLoader(self.ff2ff_timing_test, batch_size, shuffle=False, 
                                    pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=self.persistent_workers)


if __name__ == "__main__":
    dataset = FF2FF_Dataset("dataset/ff2ff_dataset_power.json", scaling='power')
    print(dataset[0])
