import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ScoreVector():
    def __init__(self,  semantic_path,scale):
        super(ScoreVector, self).__init__()
        label2vec = np.load(semantic_path).astype(np.float32)
        label2vec=label2vec/scale#/100 for CUB
        self.label2vec_dim = label2vec.shape[1]
        self.label2vec = nn.Parameter(torch.from_numpy(label2vec), requires_grad=False)
       
    def get_score_vector(self,ids):
        batch_size, num_per_batch=ids.size()
        score_vector=self.label2vec[ids.view(-1)].view(batch_size, num_per_batch,self.label2vec_dim)
        return score_vector.cuda()


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        # features_train [batch_size, num_train_examples, num_features]
        # labels_train [batch_size, num_train_examples, nKnovel]
        labels_train_transposed = labels_train.transpose(1,2)
        # labels_train_transposed [batch_size, nKnovel, num_train_examples]
        weight_novel = torch.bmm(labels_train_transposed, features_train)#1.sum up features of each class 
        # weight_novel [batch_size, nKnovel, num_features]
        
        ##2.divided by the number of examples of each class
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel
        