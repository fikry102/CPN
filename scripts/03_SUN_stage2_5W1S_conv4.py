import sys
import torch
sys.path.append("..")
from traincode_CPN import train_stage2
from testcode_CPN import test

from args_config_train import argparse_config_train
from args_config_test import argparse_config_test
gpu_id=str(torch.cuda.device_count()-1)

opt = argparse_config_train() # refer to args_config_train.py for details of auguments
opt.save_path = './experiments/sun_stage2_5W1S_conv4'
opt.embnet_pretrainedandfix = True
opt.pretrain_embnet_path = './experiments/sun_stage1_conv4/best_model.pth'
opt.gpu=gpu_id
opt.network = 'Conv4'
opt.head = 'CPN'
opt.proto_fusion = 'CPN'
# TODO: Set the appropriate semantic_path of the datasets here.
opt.semantic_path = '../../data/sun/sun_attributes.npy'
opt.nKall = 580
opt.nKbase=0#no fake base
opt.train_query = 15
opt.nTestBase=0
opt.epoch_size = 600
opt.dataset = 'SUN'
opt.avg_pool = True
# opt.nfeat = 640
opt.nfeat=64
opt.val_episode = 1000
opt.num_epoch=10
opt.episodes_per_batch = 8
opt.milestones = [5,10]
opt.lambdalr = [1.0,0.5]
opt.train_way, opt.train_shot = opt.test_way, opt.val_shot = [5,1]
train_stage2(opt)
torch.cuda.empty_cache()


opt = argparse_config_test() # refer to args_config_test.py for details of auguments
opt.load = './experiments/sun_stage2_5W1S_conv4/best_model.pth'
opt.gpu=gpu_id
opt.network = 'Conv4'
opt.head = 'CPN'
opt.proto_fusion = 'CPN'
# TODO: Set the appropriate semantic_path of the datasets here.
opt.semantic_path = '../../data/sun/sun_attributes.npy'
opt.nKall = 580
opt.dataset = 'SUN'
opt.avg_pool = True
# opt.nfeat = 640
opt.nfeat=64
opt.way, opt.shot = [5,1]
opt.query=15
opt.episode=5000
test(opt)
torch.cuda.empty_cache()