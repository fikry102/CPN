import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from models.CPN_auxilary import ScoreVector,FeatExemplarAvgBlock

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


class CPNhead(nn.Module):
    def __init__(self,opt):
        super(CPNhead, self).__init__()
        self.proto_fusion = opt.proto_fusion
        self.nFeat = opt.nfeat
        self.nKall = opt.nKall

        scale_cls = 10.0 # cosine similarity temperature parameter 
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls), requires_grad=True)
    
        print('proto_fusion:', opt.proto_fusion)
        self.favgblock = FeatExemplarAvgBlock()   
        scale =  1 if opt.dataset=='SUN' else 100    
        self.Score= ScoreVector(opt.semantic_path,scale)
        num_attribute= 102 if opt.dataset=='SUN' else 312
        self.component_proto=nn.Parameter(torch.randn(num_attribute, self.nFeat),requires_grad=True) 
        if self.proto_fusion=='CPN':
            self.weight_gen=nn.Sequential(nn.Linear(self.nFeat,1),
                        nn.Sigmoid())

    def get_classification_weights(self, Kbase_ids, Knovel_ids, features_train=None, labels_train=None):


        if features_train is None:#pre-training
            proto=F.normalize(self.component_proto,dim=-1)
            gt_base_score=self.Score.get_score_vector(Kbase_ids)
            cls_weights=torch.einsum('b m a,a c ->b m c',gt_base_score,proto)
            return cls_weights
        #else:

        if self.proto_fusion=='none':#validation (pre-training) 
            gt_support_score=self.Score.get_score_vector(Knovel_ids)
            proto=F.normalize(self.component_proto.detach(),dim=-1)
            cls_weights=torch.einsum('b m a,a c -> b m c',gt_support_score,proto)

        elif self.proto_fusion=='CPN':#meta-training, meta-testing
            gt_support_score=self.Score.get_score_vector(Knovel_ids)
            # proto=F.normalize(self.component_proto.detach(),dim=-1)
            proto=F.normalize(self.component_proto,dim=-1)
            comp_weights=torch.einsum('b m a,a c -> b m c',gt_support_score,proto)
            comp_weights=F.normalize(comp_weights,dim=-1)

            cls_weights = self.favgblock(features_train, labels_train)     
            cls_weights=F.normalize(cls_weights,dim=-1)
            
            c=self.weight_gen(comp_weights)
            cls_weights=c*comp_weights+(1-c)*cls_weights   
        return cls_weights


    def apply_classification_weights(self, features, cls_weights):
        features = F.normalize(features, p=2, dim=-1)
        
        cls_weights = F.normalize(cls_weights, p=2, dim=-1)
        
        cls_scores = self.scale_cls * torch.bmm(features, cls_weights.transpose(1,2))
        return cls_scores
    
    def forward(self, query, support, support_labels, n_way, n_shot, Kall, nKbase, **kwargs):
        tasks_per_batch = query.size(0)

        if support is not None:
            n_support = support.size(1)
            assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
            assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
        
            support_labels_one_hot = one_hot((support_labels-nKbase).view(tasks_per_batch * n_support), n_way)
            support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
        else:
            support_labels_one_hot = None
        
        # get base and novel class IDs
        Kall = Kall.long()
        Kbase_ids = (None if (nKbase==0) else Variable(Kall[:,:nKbase].contiguous(), requires_grad=False))
        Knovel_ids = Variable(Kall[:,nKbase:].contiguous(), requires_grad=False)#classes for support
        
        cls_weights = self.get_classification_weights(
                        Kbase_ids, Knovel_ids, support, support_labels_one_hot) 
        logits = self.apply_classification_weights(
                query, cls_weights)
        return logits
