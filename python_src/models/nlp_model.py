import torch as th
import numpy as np
import orjson
from torch import random
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
# import pytorch_lightning as pl

from timing_dataset import *
from hyper_params import *


class FF2FF_NLP_Model(nn.Module):
    def __init__(self, model_hparams, dataset, embedding_weights):
        super(FF2FF_NLP_Model, self).__init__()
        self.embed = nn.Embedding(embedding_weights.size(0), embedding_weights.size(1))
        self.embed.weight = nn.Parameter(embedding_weights)
        self.embed.requires_grad = False
        self.transformer = nn.TransformerEncoderLayer(d_model=30, nhead=5, dim_feedforward=MAX_LEN)
        
    
    def forward(self, seq_enc, wc_enc):
        out = self.embed(seq_enc)
        print("Embedding Out Shape:", out.shape)
        out = out.transpose(0, 1)
        out = self.transformer(out)
        return out


if __name__ == "__main__":
    batch_size = 512
    model_hparams = {

    }

    ff2ff_train_dataloader = DataLoader(ff2ff_timing_train, batch_size, shuffle=True, 
                                pin_memory=True)
    ff2ff_test_dataloader = DataLoader(ff2ff_timing_test, batch_size, shuffle=False, 
                                pin_memory=True)

    embedding_weights = th.Tensor(th.load(CHECKPOINTS_PATH+"/sns_quant_embedding-30.pt"))
    nlp_model = FF2FF_NLP_Model(model_hparams, ff2ff_timing_dataset, embedding_weights)
    
    seq, wc, timing = next(iter(ff2ff_train_dataloader))

    print(seq.shape)
    print(wc.shape)

    out = nlp_model(seq, wc)
    print(out.shape)
    print(out)
