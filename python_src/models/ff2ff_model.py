import pytorch_lightning as pl
import torch as th
from torch.utils.data import TensorDataset, DataLoader
import random
from hyper_params import MAX_LEN
from math import ceil
from collections import Counter

class FF2FF_Model(pl.LightningModule):
    def __init__(self, use_gpu=1, batch_size=512):
        super().__init__()
        self.save_hyperparameters()
    
    @staticmethod
    def MAEPLoss(target, out):
        loss =  ((target - out).abs() / target.abs()).mean() * 100.0
        return loss
    
    @staticmethod
    def RRSELoss(target, pred):
        target_mean = th.mean(target)
        s_top = th.sum((pred - target) * (pred - target))
        s_bot = th.sum((target - target_mean) * (target - target_mean))
        return (s_top / s_bot)**0.5
        
    @staticmethod
    def encode_unique(paths):
        """
        paths = tuple(map(tuple, paths))
        print(len(paths))
        path_unique = tuple(set(paths))
        path_unique = list(map(tuple, path_unique))
        print(len(path_unique))
        unique_counts = []
        for unique in path_unique:
            for 
        # print(path_unique)
        """
        paths = tuple(map(tuple, paths))
        c = dict(Counter(paths))
        # print(c)
        unique_paths = list(c.keys())
        unique_counts = list(c.values())
        # print(len(unique_paths))
        # print(len(unique_counts))
        # print(unique_paths[0])
        # print(unique_counts)
        return unique_paths, unique_counts


    def infer_on_paths_timing(self, paths):
        unique_paths, unique_counts = self.encode_unique(paths)
        paths = unique_paths

        # Sequence Encoding
        seq_enc = th.zeros((len(paths), MAX_LEN), dtype=th.long)
        wc_enc = th.zeros((len(paths), self.datamodule.ff2ff_timing_dataset.wc_len), dtype=th.float32)
        for i in range(len(paths)):
            seq_enc[i], wc_enc[i] = self.datamodule.ff2ff_timing_dataset.encode_seq(paths[i])

        # Prepare Dataset
        seq_enc_ds = TensorDataset(seq_enc, wc_enc)
        seq_enc_dl = DataLoader(seq_enc_ds, batch_size=self.hparams.batch_size, num_workers=32)
        
        trainer = pl.Trainer(gpus=self.use_gpu) if self.use_gpu >= 1 or self.use_gpu == -1 else pl.Trainer()
        
        pred = trainer.predict(self, dataloaders=seq_enc_dl)
        pred = th.cat(pred)
        pred = th.ravel(pred)

        max_idx = th.argmax(pred).cpu().item()

        real_timing_pred = self.datamodule.ff2ff_timing_dataset.denormalize_timing(pred)
        # print(real_timing_pred.shape)
        return th.max(real_timing_pred).cpu().item(), paths[max_idx]
    
    def infer_on_paths_power(self, paths):
        # unique_paths, unique_counts = self.encode_unique(paths)
        # paths = unique_paths

        # Sequence Encoding
        seq_enc = th.zeros((len(paths), MAX_LEN), dtype=th.long)
        wc_enc = th.zeros((len(paths), self.datamodule.ff2ff_timing_dataset.wc_len), dtype=th.float32)
        for i in range(len(paths)):
            seq_enc[i], wc_enc[i] = self.datamodule.ff2ff_timing_dataset.encode_seq(paths[i])

        # Prepare Dataset
        seq_enc_ds = TensorDataset(seq_enc, wc_enc)
        seq_enc_dl = DataLoader(seq_enc_ds, batch_size=self.hparams.batch_size, num_workers=32)
        
        trainer = pl.Trainer(gpus=self.use_gpu) if self.use_gpu >= 1 or self.use_gpu == -1 else pl.Trainer()
        
        pred = trainer.predict(self, dataloaders=seq_enc_dl)
        pred = th.cat(pred)
        pred = th.ravel(pred)

        max_idx = th.argmax(pred).cpu().item()

        real_timing_pred = self.datamodule.ff2ff_timing_dataset.denormalize_timing(pred)

        return th.mean(real_timing_pred).cpu().item(), paths[max_idx]

    def infer_on_circuit_timing(self, design, path_count=500):
        # Sample Paths First
        paths = design['paths']
        indices = random.choices(range(len(paths)), k=path_count)
        paths_sel = []
        for i in indices:
            # Avoid Empty Path
            if len(paths[i]) > 0:
                paths_sel.append(paths[i])
        
        return self.infer_on_paths_timing(paths_sel)
    
    def infer_on_circuit_power(self, design, path_count=500):
        # Sample Paths First
        paths = design['paths']
        indices = random.choices(range(len(paths)), k=path_count)
        paths_sel = []
        for i in indices:
            # Avoid Empty Path
            if len(paths[i]) > 0:
                paths_sel.append(paths[i])
        
        timing, max_path = self.infer_on_paths_power(paths_sel)
        
        return timing * len(paths), max_path
    
    def train_dataloader(self):
        return self.datamodule.train_dataloader(self.hparams.batch_size)
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader(self.hparams.batch_size)
