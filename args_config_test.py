import argparse

def argparse_config_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0',
                            help='choose which gpu to be used')
    parser.add_argument('--load', default='./experiments/tmp/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=2000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--nfeat', type=int, default=640,
                            help='number of feature dimension')
    parser.add_argument('--nKall', type=int, default=-1,
                            help='number of all classes')
    parser.add_argument('--nKbase', type=int, default=0,
                            help='number of base classes')
    parser.add_argument('--nTestBase', type=int, default=0,
                            help='number of query examples per testing class')
    parser.add_argument('--epoch_size', type=int, default=1000,
                            help='number of episodes per epoch')
    parser.add_argument('--avg-pool', default=False, action='store_true',
                            help='whether to do average pooling in the last layer of ResNet models')
    parser.add_argument('--network', type=str, default='ResNet12',
                            help='choose which embedding network to use. ResNet12, Conv4')
    parser.add_argument('--dataset', type=str, default='CUB',
                            help='choose dataset to use. CUB, SUN')
    parser.add_argument('--semantic_path', type=str, default='No semantic to be used',
                            help='semantic path for current dataset.')

    args = parser.parse_known_args()[0]
    return args

