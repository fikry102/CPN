import sys
import torch
sys.path.append("..")
from traincode_CPN import train_stage1
from args_config_train import argparse_config_train
gpu_id=str(torch.cuda.device_count()-1)

opt = argparse_config_train() # refer to args_config_train.py for details of auguments
# TODO: Set the appropriate semantic_path of the datasets here.
opt.semantic_path = '../../data/sun/sun_attributes.npy'
opt.save_path = './experiments/sun_stage1_conv4'
opt.gpu = gpu_id
opt.network = 'Conv4'
opt.head = 'CPN'
opt.proto_fusion = 'none'
opt.nKall = 580
opt.nKbase = 580
opt.train_query = 0
opt.nTestBase = 32
opt.epoch_size = 1000
opt.dataset='SUN'
opt.avg_pool = True
# opt.nfeat = 640
opt.nfeat=64
opt.val_episode = 1000
opt.num_epoch = 30
opt.episodes_per_batch = 4
opt.milestones = [20]
opt.train_way, opt.train_shot = [0,0]
opt.test_way, opt.val_shot = [5,1]
train_stage1(opt)
torch.cuda.empty_cache()