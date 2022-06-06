from pytorch_lightning.core.hooks import ModelHooks
from torch.functional import Tensor
from torch.optim.adam import Adam
from transformers import BertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
import torch.nn as nn
import pytorch_lightning as pl
import torch as th
from pytorch_lightning.tuner.tuning import Tuner
from timing_dataset import FF2FF_Data_Module
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
from hyper_params import MAX_LEN
import orjson
from torch.utils.data import TensorDataset, DataLoader
import sys, os
from ff2ff_model import FF2FF_Model
from model_helper import maep, rrse, mep, save_test_result, corre_coeff
import warnings

warnings.filterwarnings('ignore')
import itertools
import torch.multiprocessing
from torchsummary import summary

torch.multiprocessing.set_sharing_strategy('file_system')
import json

count = 0

POWER_CHK_POINT = "saved_models/mobilebert_power_final.pt"
TIMING_CHK_POINT = "saved_models/mobilebert_timing_final.pt"


# final_model =

class Circuitformer(FF2FF_Model):
    def __init__(self, num_attention_heads=2,
                 intermediate_size=256,
                 num_hidden_layers=1,
                 logits_size=1,
                 embedding_dim=30,
                 vocab_size=50,
                 batch_size=512,
                 iteration=0,
                 patient=5,
                 val_percent=30):
        super().__init__()
        self.use_gpu = -1
        self.train_loss = []
        self.val_loss = []

        self.bert_model_config = MobileBertConfig(vocab_size=vocab_size,
                                                  num_attention_heads=num_attention_heads,
                                                  hidden_size=embedding_dim,
                                                  num_hidden_layers=num_hidden_layers,
                                                  intermediate_size=intermediate_size)

        self.bert_model_config.num_labels = logits_size
        self.bert_model = MobileBertForSequenceClassification(self.bert_model_config)

        self.out_linear = nn.Linear(logits_size, 1)
        # self.logits_size = logits_size
        self.save_hyperparameters()

    def forward(self, seq_enc):
        attn_mask = seq_enc != 0

        out = self.bert_model(input_ids=seq_enc, attention_mask=attn_mask).logits

        if self.hparams.logits_size != 1:
            out = self.out_linear(out)

        return out

    def training_step(self, batch, batch_idx):
        X_train, wc_enc, target = batch

        out = self(X_train)
        out = th.ravel(out)

        # print(target, out)
        loss = self.MAEPLoss(target, out)
        # print("Training Log:", self.current_epoch, batch_idx, loss.item())
        self.train_loss.append({"epoch": self.current_epoch,
                                "batch_idx": batch_idx,
                                "batch_len": X_train.shape[0],
                                "loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        X_train, wc_enc, target = batch

        out = self(X_train)
        out = th.ravel(out)

        loss = self.MAEPLoss(target, out)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.val_loss.append({"epoch": self.current_epoch,
                              "batch_idx": batch_idx,
                              "batch_len": X_train.shape[0],
                              "loss": loss.item()})

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001)

    def predict_step(self, batch, batch_idx):
        X_train, wc_enc = batch
        # print(X_train[0], wc_enc[0])
        out = self(X_train)
        return out


def train_power(hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 256,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 5
}, checkpoint=POWER_CHK_POINT):
    # Initialize Model and Dataset
    model = Circuitformer(**hparams)
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", scaling='timing')
    # dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power', val_set_percentage=30)
    model.datamodule = dm

    # Decide the Batchsize
    trainer = pl.Trainer(gpus=1, auto_scale_batch_size=True)
    tuner = Tuner(trainer)

    new_batch_size = tuner.scale_batch_size(model)
    print("Training on Batch Size: ", new_batch_size)
    model.hparams.batch_size = new_batch_size

    # Start Training
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1, patience=hparams['patient'], verbose=False, mode="min")
    # trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback])
    trainer = pl.Trainer(gpus=1, max_epochs=256)
    trainer.fit(model)

    # Save Model
    trainer.save_checkpoint(checkpoint)
    json.dump({"train": model.train_loss, "val": model.val_loss},
              open("eval/power_training_loss_log_{}.json".format(count), "w"))


def train_timing(hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 128,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 100.0,
    "val_percent": 30
}, checkpoint=TIMING_CHK_POINT):
    # Initialize Model and Dataset
    model = Circuitformer(**hparams)
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", val_set_percentage=hparams['val_percent'],
                           scaling='timing')
    # dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power')
    model.datamodule = dm

    # Decide the Batchsize
    trainer = pl.Trainer(gpus=1, auto_scale_batch_size=True)
    tuner = Tuner(trainer)

    new_batch_size = tuner.scale_batch_size(model)
    print("Training on Batch Size: ", new_batch_size)
    model.hparams.batch_size = new_batch_size

    # Start Training
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1, patience=hparams['patient'], verbose=False, mode="min")
    # trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback])
    trainer = pl.Trainer(gpus=1, max_epochs=256)
    trainer.fit(model)

    # Save Model
    trainer.save_checkpoint(checkpoint)

    json.dump({"train": model.train_loss, "val": model.val_loss},
              open("eval/timing_training_loss_log_{}.json".format(count), "w"))


def train_timing_final():
    hparams = {
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "logits_size": 1,
        "embedding_dim": 128,
        "vocab_size": 50,
        "batch_size": 512,
        "patient": 100.0
    }

    # Initialize Model and Dataset
    model = Circuitformer(**hparams)
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", scaling='timing', val_set_percentage=1)
    # dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power')
    model.datamodule = dm

    # Decide the Batchsize
    trainer = pl.Trainer(gpus=1, auto_scale_batch_size=True)
    tuner = Tuner(trainer)

    new_batch_size = tuner.scale_batch_size(model)
    print("Training on Batch Size: ", new_batch_size)
    model.hparams.batch_size = 128

    # Start Training
    trainer = pl.Trainer(gpus=1, max_epochs=256)
    trainer.fit(model)

    # Save Model
    trainer.save_checkpoint(TIMING_CHK_POINT)


def train_power_final():
    hparams = {
        "num_attention_heads": 2,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "logits_size": 1,
        "embedding_dim": 128,
        "vocab_size": 50,
        "batch_size": 512,
        "patient": 100.0
    }

    # Initialize Model and Dataset
    model = Circuitformer(**hparams)
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power', val_set_percentage=1)
    # dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power')
    model.datamodule = dm

    # Decide the Batchsize
    trainer = pl.Trainer(gpus=1, auto_scale_batch_size=True)
    tuner = Tuner(trainer)

    new_batch_size = tuner.scale_batch_size(model)
    print("Training on Batch Size: ", new_batch_size)
    model.hparams.batch_size = 128

    # Start Training
    trainer = pl.Trainer(gpus=1, max_epochs=256)
    trainer.fit(model)

    # Save Model
    trainer.save_checkpoint(POWER_CHK_POINT)


def test_timing(hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 128,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 100.0
}, basename="mobilebert_timing_final", checkpoint=TIMING_CHK_POINT):
    new_model = Circuitformer.load_from_checkpoint(checkpoint_path=checkpoint)
    new_model.hparams.batch_size = 512
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_684_beta.json", scaling='timing')
    new_model.datamodule = dm

    # summary(new_model, (1, 512))
    # exit()

    real_timing = []
    pred_timing = []
    for did in range(42):
        with open("dataset/merged_test_set/design{}.json".format(did), 'r') as json_file:
            design = orjson.loads(json_file.read())

            timing, max_path = new_model.infer_on_circuit_timing(design, path_count=8192)
            total_num_paths = design['paths']

            print("Real Timing: ", design['timing'], "Predict: ", timing)
            real_timing.append(design['timing'])
            pred_timing.append(timing)
        print("{}/42 Designs Processed.".format(did))

    real_timing = th.Tensor(real_timing)
    pred_timing = th.Tensor(pred_timing)
    print("RRSE: ", rrse(pred_timing, real_timing).item())
    print("MAEP: ", maep(pred_timing, real_timing).item())
    print("MEP: ", mep(pred_timing, real_timing).item())
    print("r:", corre_coeff(pred_timing, real_timing))
    save_test_result(basename, new_model.hparams, real_timing, pred_timing)


def test_power(hparams={
    "num_attention_heads": 2,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "logits_size": 1,
    "embedding_dim": 256,
    "vocab_size": 50,
    "batch_size": 512,
    "patient": 5
}, basename="mobilebert_power_final", checkpoint=POWER_CHK_POINT):
    new_model = Circuitformer.load_from_checkpoint(checkpoint_path=checkpoint)
    new_model.hparams.batch_size = 512
    dm = FF2FF_Data_Module("dataset/ff2ff_dataset_power.json", scaling='power')
    new_model.datamodule = dm

    real_timing = []
    pred_timing = []
    for did in range(42):
        # with open("dataset/merged_test_set/design_t{}.json".format(did), 'r') as json_file:
        with open("dataset/merged_test_set/design{}.json".format(did), 'r') as json_file:
            design = orjson.loads(json_file.read())

            timing, max_path = new_model.infer_on_circuit_power(design, path_count=8192)
            print("Real Timing: ", design['power'], "Predict: ", timing)
            real_timing.append(design['power'])
            pred_timing.append(timing)
        print("{}/42 Designs Processed.".format(did + 1))

    real_timing = th.Tensor(real_timing)
    pred_timing = th.Tensor(pred_timing)
    print("RRSE: ", rrse(pred_timing, real_timing).item())
    print("MAEP: ", maep(pred_timing, real_timing).item())
    print("MEP: ", mep(pred_timing, real_timing).item())
    print("r:", corre_coeff(pred_timing, real_timing))
    save_test_result(basename, new_model.hparams, real_timing, pred_timing)


if __name__ == "__main__":
    train_power_final()
