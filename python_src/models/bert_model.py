from torch.optim.adam import Adam
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt
import transformers
from transformers.optimization import Adafactor, AdafactorSchedule
from prettytable import PrettyTable

from timing_dataset import *
from lr_model import MAEPLoss
import time
scaler = torch.cuda.amp.GradScaler()

PLOT = True
batch_size = 1024

class FF2FF_Bert_Model(pl.LightningModule):
    def __init__(self, modelhparams, dataset, embedding_weights):
        super(FF2FF_Bert_Model, self).__init__()

        self.modelhparams = modelhparams

        self.dataset = dataset

        self.embed = nn.Embedding(embedding_weights.size(0), embedding_weights.size(1))
        self.embed.weight = nn.Parameter(embedding_weights)
        self.embed.requires_grad = False

        self.bert_model_config = BertConfig(vocab_size=dataset.wc_len, 
                hidden_size=embedding_weights.size(1),
                num_attention_heads=modelhparams["num_attention_heads"], 
                num_hidden_layers=modelhparams["num_hidden_layers"],
                intermediate_size=modelhparams["intermediate_size"],
                hidden_act=modelhparams["hidden_act"],
                max_position_embeddings=modelhparams["max_position_embeddings"])
        
        self.bert_model_config.num_labels = modelhparams["logits_size"]
        self.bert_model = BertForSequenceClassification(self.bert_model_config)

        self.out_linear = nn.Linear(modelhparams["logits_size"], 1)
    
    def forward(self, seq_enc):
        attn_mask = seq_enc != 0

        out = None

        if self.modelhparams["custom_embedding"]:
            out = self.embed(seq_enc)

            out = self.bert_model(inputs_embeds=out, attention_mask=attn_mask).logits

        else:
            out = self.bert_model(input_ids=seq_enc, attention_mask=attn_mask).logits

        if self.modelhparams["logits_size"] != 1:
            out = self.out_linear(out)

        return out

if __name__ == "__main__":
    model_hparams = {
        "num_attention_heads": 3,
        "num_hidden_layers": 1,
        "intermediate_size": 256,
        "hidden_act": 'gelu',
        "max_position_embeddings": 512,
        "logits_size": 1,
        "custom_embedding": True,
        "epochs": 500
    }

    """
    ff2ff_train_dataloader = DataLoader(ff2ff_timing_train, batch_size, shuffle=True, 
                                pin_memory=True)
    ff2ff_test_dataloader = DataLoader(ff2ff_timing_test, batch_size, shuffle=False, 
                                pin_memory=True)

    X_train, X_train_wc, y_train = next(iter(ff2ff_train_dataloader))
    X_test, X_test_wc, y_test = next(iter(ff2ff_test_dataloader))
    """
    embedding_weights = th.Tensor(th.load(CHECKPOINTS_PATH+"/sns_quant_embedding-30.pt"))
    model = FF2FF_Bert_Model(model_hparams, ff2ff_timing_dataset, embedding_weights)

    print("Model Parameters: ", count_parameters(model))

    scores = score_model(FF2FF_Bert_Model, model_hparams, ff2ff_timing_dataset, embedding_weights)
    print(th.mean(scores))
    exit()

    model = model.to('cuda:0')
    X_train = X_train.to('cuda:0')
    X_train_wc = X_train_wc.to('cuda:0')
    X_test = X_test.to('cuda:0')
    X_test_wc = X_test_wc.to('cuda:0')
    y_train = y_train.to('cuda:0')
    y_test = y_test.to('cuda:0')

    num_epochs = 500
    for epoch in range(num_epochs):
        target = y_train

        # forward
        out = model(X_train)
        out = th.ravel(out)

        loss =  MAEPLoss(target, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
    
    model.eval()
    with th.no_grad():
        predict = model(X_train)
        test_pred = model(X_test)
    predict = predict.cpu().data.numpy()
    
    print("Test Loss:", MAEPLoss(th.ravel(test_pred), y_test))
    test_pred = test_pred.cpu().data.numpy()

    th.save(model, CHECKPOINTS_PATH+"/bert_model.pt")

    if PLOT:
        plt.scatter(y_train.cpu().numpy(), predict)
        plt.scatter(y_test.cpu().numpy(), test_pred)
        plt.plot([0, 1], [0, 1])

        plt.title("BERT Model")
        plt.xlabel("Real Timing (ns)")
        plt.ylabel("Prediction (ns)")
        plt.legend(["target", "train", "test"])

        plt.show()