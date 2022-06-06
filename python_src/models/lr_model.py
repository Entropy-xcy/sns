import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from timing_dataset import *
import warnings
warnings.filterwarnings('ignore')
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ff2ff_model import FF2FF_Model
from model_helper import maep, rrse, mep, save_test_result, corre_coeff

CHK_POINT = "saved_models/lr_model.pt"

# Linear Regression Model
class FF2FF_Linear_Regression_Model(FF2FF_Model):
    def __init__(self, vocab_size=50, batch_size=1024):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, 1)
        self.use_gpu = 0
        self.save_hyperparameters()

    @staticmethod
    def MAEPLoss(target, out):
        loss =  ((target - out).abs() / target.abs()).mean() * 100.0
        return loss

    def forward(self, x):
        sequence_enc_ret, wc_enc = x
        out = self.linear1(wc_enc)
        return out
    
    def training_step(self, batch, batch_idx):
        sequence_enc_ret, wc_enc, target = batch

        out = self((sequence_enc_ret, wc_enc))
        out = th.ravel(out)
        
        # print(target, out)
        loss = self.MAEPLoss(target, out)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequence_enc_ret, wc_enc, target = batch

        out = self((sequence_enc_ret, wc_enc))
        out = th.ravel(out)

        loss = self.MAEPLoss(target, out)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        X_train, wc_enc = batch
        
        out = self((X_train, wc_enc))
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.05)
    

def train():
    model = FF2FF_Linear_Regression_Model(vocab_size=50)
    dm = FF2FF_Timing_Data_Module("./dataset/ff2ff_dataset_684_beta.json")
    model.datamodule = dm

    early_stop_callback = EarlyStopping(
                    monitor="val_loss", 
                    min_delta=2, patience=500, verbose=False, mode="min")
    trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=5000)
    trainer.fit(model)
    trainer.save_checkpoint(CHK_POINT)

def test():
    new_model = FF2FF_Linear_Regression_Model.load_from_checkpoint(checkpoint_path=CHK_POINT)
    dm = FF2FF_Timing_Data_Module("./dataset/ff2ff_dataset_684_beta.json")
    new_model.datamodule = dm
    new_model.hparams.batch_size = 32768

    real_timing = []
    pred_timing = []
    for did in range(54):
        with open("dataset/timing_test_set/design{}.json".format(did), 'r') as json_file:
            design = orjson.loads(json_file.read())
    
            timing, max_path = new_model.infer_on_circuit(design, path_count=32768)
            print("Real Timing: ", design['timing'], "Predict: ", timing)
            real_timing.append(design['timing'])
            pred_timing.append(timing)
        print("{}/54 Designs Processed.".format(did))

    real_timing = th.Tensor(real_timing)
    pred_timing = th.Tensor(pred_timing)
    print("RRSE: ", rrse(pred_timing, real_timing).item())
    print("MAEP: ", maep(pred_timing, real_timing).item())
    print("MEP: ", mep(pred_timing, real_timing).item())
    print("r:", corre_coeff(pred_timing, real_timing))
    save_test_result("lr_timing", new_model.hparams, real_timing, pred_timing)

if __name__ == "__main__":
    train()
    test()
