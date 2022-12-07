import sys
import torch
sys.path.append("..")
from traincode_CPN import train_stage1
from args_config_train import argparse_config_train
gpu_id=str(torch.cuda.device_count()-1)

opt = argparse_config_train() # refer to args_config_train.py for details of auguments
opt.save_path = './experiments/cub_stage1'
opt.semantic_path = '../../data/cub/cub_attributes.npy'
opt.gpu = gpu_id
opt.network = 'ResNet12'
opt.head = 'CPN'
opt.proto_fusion = 'none'
opt.nKall = 100
opt.nKbase = 100
opt.train_query = 0
opt.nTestBase = 32
opt.epoch_size = 1000
opt.dataset = 'CUB'
opt.avg_pool = True
opt.nfeat = 640
opt.val_episode = 1000
opt.num_epoch = 30
opt.episodes_per_batch = 4
opt.milestones = [20]
opt.train_way, opt.train_shot = [0,0]
opt.test_way, opt.val_shot = [5,1]
train_stage1(opt) 
torch.cuda.empty_cache()

