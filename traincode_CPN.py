# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import torch

from models.CPN_head import CPNhead
from models.ResNet12 import resnet12
from models.Conv4 import conv4

from utils import set_gpu, Timer, count_accuracy, check_dir, log

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

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
        dataset_train = CUB(phase='train')
        dataset_val = CUB(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'SUN':
        from data.SUN import SUN, FewShotDataloader
        dataset_train = SUN(phase='train')
        dataset_val = SUN(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

def train_stage1(opt):
    assert(opt.embnet_pretrainedandfix==False)
    assert(opt.train_way == 0 and opt.nKall != -1)
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=opt.nKbase,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=opt.nTestBase, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * opt.epoch_size, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=1.0, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: opt.lambdalr[0] if e < opt.milestones[0] else opt.lambdalr[1]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch + 1):
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
       

        train_losses = []
        #opt.epoch_size*opt.episodes_per_batch
        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]
            emb_support = labels_support = None

            nKbase = nKbase.squeeze()[0].item()

            train_n_query = opt.train_way * opt.train_query + opt.nTestBase
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot, 
            Kall=Kall, nKbase=nKbase)

            cls_way = opt.nKall

            loss = x_entropy(logit_query.reshape(-1, cls_way), labels_query.reshape(-1))

            acc = count_accuracy(logit_query.reshape(-1, cls_way), labels_query.reshape(-1))
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())
            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, Kall=Kall, 
            nKbase=0)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)#confidence interval

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
        lr_scheduler.step()

def train_stage2(opt):
    assert(opt.embnet_pretrainedandfix==True)
    assert(opt.train_way > 0)
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=opt.nKbase,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=opt.nTestBase, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * opt.epoch_size, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)

    # Load saved model checkpoints
    saved_models = torch.load(opt.pretrain_embnet_path)
    embedding_net.load_state_dict(saved_models['embedding'])
    del saved_models['head']['scale_cls'] #introduce \tau_2
    cls_head.load_state_dict(saved_models['head'], strict=False)
    if opt.embnet_pretrainedandfix==True:
        embedding_net.eval()
        optimizer = torch.optim.SGD([{'params': cls_head.parameters()}], lr=1.0, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    # else:
    #     optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
    #                             {'params': cls_head.parameters()}], lr=1.0, momentum=0.9, \
    #                                     weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: opt.lambdalr[0] if e < opt.milestones[0] else opt.lambdalr[1]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, opt.num_epoch + 1):
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        if opt.embnet_pretrainedandfix==True:
            cls_head.train()
        # else:
        #     _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            with torch.no_grad():
                data_support, labels_support, data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]
                train_n_support = opt.train_way * opt.train_shot
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
                
                
                nKbase = nKbase.squeeze()[0].item()

                train_n_query = opt.train_way * opt.train_query + opt.nTestBase
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)
                
                
            
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot, 
            Kall=Kall, nKbase=nKbase)

            cls_way = opt.train_way#meta-training

            loss = x_entropy(logit_query.reshape(-1, cls_way), labels_query.reshape(-1))    
            acc = count_accuracy(logit_query.reshape(-1, cls_way), labels_query.reshape(-1))
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())
            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        if opt.embnet_pretrainedandfix==True:
            cls_head.eval() 
        else:
            _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, Kall, nKbase = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query
            with torch.no_grad():
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_support = emb_support.reshape(1, test_n_support, -1)

                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, Kall=Kall, nKbase=0)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        lr_scheduler.step()


        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))