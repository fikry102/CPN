# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
from models.ResNet12 import resnet12
from models.Conv4 import conv4
from models.CPN_head import CPNhead

from utils import set_gpu, count_accuracy, log

import numpy as np
import os

def get_model(opt):
    # Choose the embedding network
    if opt.network == 'ResNet12':
        network = resnet12(avg_pool=opt.avg_pool, drop_rate=0.1, dropblock_size=5).cuda()
        # network = torch.nn.DataParallel(network, device_ids=[0, 1])
    elif opt.network == 'Conv4':
        network = conv4(in_planes=3, userelu=False, num_stages=4).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if opt.head == 'CPN':
        cls_head = CPNhead(opt).cuda()   
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'CUB':
        from data.CUB import CUB, FewShotDataloader
        dataset_test = CUB(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'SUN':
        from data.SUN import SUN, FewShotDataloader
        dataset_test = SUN(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

def test(opt):
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    
    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    
    # Load saved model checkpoints
    if opt.load != 'pretrian-features':
        saved_models = torch.load(opt.load)
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        cls_head.load_state_dict(saved_models['head'])
        cls_head.eval()
    
    # Evaluate on test set
    test_accuracies = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query
        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, n_query, -1)
            
        logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, Kall=Kall, nKbase=nKbase)

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)
        
        if i % 50 == 0:
            log(log_file_path, 'Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))
