import sys
import torch
sys.path.append("..")
from traincode_CPN import train_stage2
from testcode_CPN import test

from args_config_train import argparse_config_train
from args_config_test import argparse_config_test
gpu_id=str(torch.cuda.device_count()-1)

opt = argparse_config_train() # refer to args_config_train.py for details of auguments
opt.save_path = './experiments/cub_stage2_5W5S'
opt.embnet_pretrainedandfix = True
opt.pretrain_embnet_path = './experiments/cub_stage1/best_model.pth'
opt.gpu = gpu_id
opt.network = 'ResNet12'
opt.head = 'CPN'
opt.proto_fusion = 'CPN'
# TODO: Set the appropriate semantic_path of the datasets here.
opt.semantic_path = '../../data/cub/cub_attributes.npy'
opt.nKall = 100
opt.nKbase = 0
opt.train_query=15
opt.nTestBase = 0
opt.epoch_size = 600
opt.dataset = 'CUB'
opt.avg_pool = True
opt.nfeat = 640
opt.val_episode = 1000
opt.num_epoch = 10
opt.episodes_per_batch = 8
opt.milestones = [5]
opt.lambdalr = [1.0,0.5]
opt.train_way, opt.train_shot = opt.test_way, opt.val_shot = [5,5]
train_stage2(opt)
torch.cuda.empty_cache()


opt = argparse_config_test() # refer to args_config_test.py for details of auguments
opt.load = './experiments/cub_stage2_5W5S/best_model.pth'
opt.gpu = gpu_id
opt.network = 'ResNet12'
opt.head = 'CPN'
opt.proto_fusion='CPN'
# TODO: Set the appropriate semantic_path of the datasets here.
opt.semantic_path = '../../data/cub/cub_attributes.npy'
opt.nKall = 100
opt.dataset = 'CUB'
opt.avg_pool = True
opt.nfeat = 640
opt.way, opt.shot = [5,5]
opt.query=15
opt.episode=5000
test(opt)
torch.cuda.empty_cache()