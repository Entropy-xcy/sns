import torch

val_set_percentage = 30
# batch_size = 4096
gpu = "cuda:0"
use_gpu = True
MAX_LEN = 512
lr_epochs = 512
nn_epochs = 512
num_workers = 20
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
# print("Using device", device)
DATASET_PATH = "./dataset"
CHECKPOINTS_PATH = "./saved_models"
