import logging
import math
from selectors import EpollSelector
import sys
from argparse import Namespace
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple, List
from timeit import default_timer as timer
sys.path.append(str(Path(__file__).resolve().parent.parent))


import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils import data
from art.prop import AndProp
from art.bisecter import Bisecter
from art import exp, utils
from art.repair_moudle import Netsum
from cifar10_utils import CifarProp, Cifar_feature_patch_model_big, Resnet_model, Vgg_model
# from mnist.mnist_utils import MnistNet_CNN_small, MnistNet_FNN_big, MnistNet_FNN_small, MnistProp, Mnist_feature_patch_model
# from mnist.u import MnistNet, MnistFeatureProp
torch.manual_seed(10668091966382305352)
device = torch.device(f'cuda:2')



CIFAR_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'cifar10'
CIFAR_NET_DIR = Path(__file__).resolve().parent.parent / 'model' / 'cifar10'
RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'cifar10' / 'big'
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'cifar10_big_format'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)




class CifarArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        self.add_argument('--net', type=str, choices=['vgg19, ResNet18'], default='Vgg19', 
                          help='network architecture')


        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        self.add_argument('--repair_number', type=int, default=50,
                          help='the number of repair datas')
        self.add_argument('--repair_batchsize', type=int, default=1,
                            help='the batchsize of repair datas')
        self.add_argument('--repair_location', type=str, default='feature',
                          choices=['feature'], help='repair the feature')

        
        # the combinational form of support and patch net
        # self.add_argument('--reassure_support_and_patch_combine',type=bool, default=False,
        #                 help='use REASSURE method to combine support and patch network')

        self.add_argument('--repair_radius',type=float, default=8, 
                          help='the radius bit of repairing datas or features')
        self.add_argument('--loose_bound',type=float, default=0.1,
                            help='the loose bound of the feature bound, which is multiplied by the radius')

        # training
        self.add_argument('--divided_repair', type=int, default=1, help='batch size for training')


        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'SmoothL1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')


        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')
        self.add_argument('--train_datasize', type=int, default=10000, 
                          help='dataset size for training')
        self.add_argument('--test_datasize', type=int, default=2000,
                          help='dataset size for test')

        # querying a verifier
        self.add_argument('--max_verifier_sec', type=int, default=300,
                          help='allowed time for a verifier query')
        self.add_argument('--verifier_timeout_as_safe', type=literal_eval, default=True,
                          help='when verifier query times out, treat it as verified rather than unknown')

        self.set_defaults(exp_fn='test_goal_safety', use_scheduler=True)
        return

    def setup_rest(self, args: Namespace):
        super().setup_rest(args)

        def ce_loss(outs: Tensor, labels: Tensor):
            softmax = nn.Softmax(dim=1)
            ce = nn.CrossEntropyLoss()
            return ce(softmax(outs), labels)
            # return ce(outs, labels)

        def Bce_loss(outs: Tensor, labels: Tensor):
            bce = nn.BCEWithLogitsLoss()
            return bce(outs, labels)

        args.accuracy_loss = {
            'L1': nn.L1Loss(),
            'MSE': nn.MSELoss(),
            'CE': ce_loss
        }[args.accuracy_loss]


        return
    pass

class CifarPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, train: bool, device, 
            net,
            repairnumber = None,
            trainnumber = None, testnumber = None, radius = 0,
            is_test_accuracy = False, 
            is_attack_testset_repaired = False, 
            is_attack_repaired = False,
            is_origin_data = False):
        '''
        trainnumber: 训练集数据量
        testnumber: 测试集数据量
        radius: 修复数据的半径
        is_test_accuracy: if True, 检测一般测试集的准确率
        is_attack_testset_repaired: if True, 检测一般被攻击测试集的准确率
        is_attack_repaired: if True, 检测被攻击数据的修复率
        三个参数只有一个为True
        '''
        #_attack_data_full
        suffix = 'train' if train else 'test'
        if train:
            fname = f'train_norm.pt'  # note that it is using original data
            # fname = f'{suffix}_norm00.pt'
            # mnist_train_norm00_dir = "/pub/data/chizm/"
            # combine = torch.load(mnist_train_norm00_dir+fname, device)
            combine = torch.load(Path(CIFAR_DATA_DIR, fname), device)
            inputs, labels = combine 
            inputs = inputs[:trainnumber]
            labels = labels[:trainnumber]
        else:
            if is_test_accuracy:
                fname = f'test_norm.pt'
                combine = torch.load(Path(CIFAR_DATA_DIR, fname), device)
                inputs, labels = combine
                # inputs = inputs[:testnumber]
                # labels = labels[:testnumber]
            elif is_origin_data:
                fname = f'origin_data_{net}_{radius}.pt'
                combine = torch.load(Path(CIFAR_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]
            
            elif is_attack_testset_repaired:
                fname = f'test_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(CIFAR_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_attack_repaired:
                fname = f'train_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(CIFAR_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]

            # clean_inputs, clean_labels = clean_combine
            # inputs = torch.cat((inputs[:testnumber], clean_inputs[:testnumber] ), dim=0)
            # labels = torch.cat((labels[:testnumber], clean_labels[:testnumber] ),  dim=0)
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def eval_test(net, set: CifarPoints, bitmap: Tensor = None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        if bitmap is None:
            outs = net(set.inputs)
        else:
            outs = net(set.inputs, bitmap)
        predicted = outs.argmax(dim=1)
        correct = (predicted == set.labels).sum().item()
        # ratio = correct / len(testset)
        ratio = correct / len(set.inputs)

    return ratio


def repair_cifar(args: Namespace, weight_clamp = False)-> Tuple[int, float, bool, float]:


    # originalset = torch.load(Path(MNIST_DATA_DIR, f'origin_data_{args.repair_radius}.pt'), device)
    # originalset = MnistPoints(inputs=originalset[0], labels=originalset[1])
    originalset = CifarPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_origin_data=True)
    repairset = CifarPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_attack_repaired=True)
    trainset = CifarPoints.load(train=True, device=device, net=args.net, repairnumber=args.repair_number, trainnumber=args.train_datasize)
    testset = CifarPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize,is_test_accuracy=True)
    attack_testset = CifarPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize, radius=args.repair_radius,is_attack_testset_repaired=True)
    # if args.net == 'CNN_small':
    #     net = MnistNet_CNN_small(dom=args.dom)
    # elif args.net == 'FNN_big':
    #     net = MnistNet_FNN_big(dom=args.dom)
    # elif args.net == 'FNN_small':
    #     net = MnistNet_FNN_small(dom=args.dom)
    # net.to(device)
    
    if args.net == 'vgg19':
        net = Vgg_model(dom=args.dom)
        fname = f'vgg19.pth'
        fpath = Path(CIFAR_NET_DIR, fname)
    elif args.net == 'resnet18':
        net = Resnet_model(dom=args.dom)
        fname = f'resnet18.pth'
        fpath = Path(CIFAR_NET_DIR, fname)
    

    net.load_state_dict(torch.load(fpath))
    net.to(device)

    net.eval()
    # test init accuracy
    logging.info(f'--Test repair set accuracy {eval_test(net, repairset)}')
    logging.info(f'--Test original set accuracy {eval_test(net, originalset)}')
    logging.info(f'--Test test set accuracy {eval_test(net, testset)}')
    logging.info(f'--Test attack test set accuracy {eval_test(net, attack_testset)}')
    logging.info(f'--Test train set accuracy {eval_test(net, trainset)}')
    # judge the batch_inputs is in which region of property
    def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor):
        '''
        in_lb: n_prop * input
        in_ub: n_prop * input
        batch_inputs: batch * input
        '''
        with torch.no_grad():
        
            batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
            if len(in_lb.shape) == 2:
                batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
                is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
            elif len(in_lb.shape) == 4:
                if in_lb.shape[0] > 200:
                    is_in_list = []
                    for i in range(batch_inputs_clone.shape[0]):
                        batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                        is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                        is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                        is_in_list.append(is_in_datai)
                    is_in = torch.stack(is_in_list, dim=0)
                else:
                    batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                    is_in = (batch_inputs_clone >= (in_lb - 1e-4)) & (batch_inputs_clone <= (in_ub + 1e-4))
                    is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop


            
            
            
            # convert to bitmap
            bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device).to(torch.uint8)
            # is in is a batch * in_bitmap.shape[0] tensor, in_bitmap.shape[1] is the number of properties
            # the every row of is_in is the bitmap of the input which row of in_bitmap is allowed
            bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
            if bitmap_i.shape[0] != 0:
                bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
            else:
                pass



            return bitmap

    # sample the repairset datas from the radius with original data

    def sample_datas_from_radius(net,
                                 dataset: CifarPoints, 
                                 radius = args.repair_radius/255,
                                 datanumber = 1e3):


        # step2 sample the data using pgd attack in the radius of the original data
        images = dataset.inputs.clone().detach()
        labels = dataset.labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        # adv_images = images.clone().detach()

        data_number_record_list = []

        full_adv_images_cat = []
        fgsm_ith = 0
        pgd_ith = 0
        for curr_image, label in zip(images, labels):
            fgsm_ith+=1
            logging.info(f'--FGSM sample {fgsm_ith} image')
            curr_image = curr_image.unsqueeze(0)
            label = label.unsqueeze(0)
            attack_datas = []
            num = 0
            step = 0
            # curr_image.requires_grad = True
            adv_curr_image = curr_image.clone().detach()
            
            # step1: random start at every adv_curr_image
            while_number = 0
            while True:
                adv_curr_image = adv_curr_image + torch.empty_like(adv_curr_image).uniform_(-radius*0.1, radius*0.1)
                # adv_curr_image = torch.clamp(adv_curr_image, min=0, max=1).detach()
                adv_curr_image = adv_curr_image.detach()
                adv_curr_image.requires_grad = True
                outputs = net(adv_curr_image)
                cost = loss(outputs, label)
                grad = torch.autograd.grad(cost, adv_curr_image, retain_graph=False, create_graph=False)[0]
                
                adv_curr_image = adv_curr_image.detach() + radius*grad.sign()
                delta = torch.clamp(adv_curr_image - curr_image, min=-radius, max=radius)
                # adv_curr_image = torch.clamp(curr_image + delta, min=0, max=1).detach()
                adv_curr_image = (curr_image + delta).detach()

                outputs = net(adv_curr_image)
                _, predicted = torch.max(outputs.data, 1)
                if label != predicted:
                    attack_datas.append(adv_curr_image)
                    num+=1
                else:
                    pass
                
                
                if step <= 50:
                    step+=1
                    continue
            
                elif step > 50:
                    # cat the attacked data
                    if attack_datas == []:
                        while_number+=1
                        if while_number <= 5:
                            adv_curr_image = adv_curr_image + torch.empty_like(adv_curr_image).uniform_(-radius, radius).detach()
                            # adv_curr_image = torch.clamp(adv_curr_image, min=0, max=1).detach()
                            step = 0
                            continue
                        else:
                            # if the attack_datas is empty, we use the repair data as the attack data
                            logging.info(f'--The attack_datas is empty, we use the repair data as the attack data')
                            adv_curr_image = repairset.inputs[fgsm_ith-1].unsqueeze(0).clone()
                            attack_datas.append(adv_curr_image)
                            break

                    curr_adv_images_cat = torch.cat(attack_datas)
                    # check every data is distinct
                    curr_adv_images_cat = torch.unique(curr_adv_images_cat, dim=0)
                    
                    
                    break

            # step2: pgd attack from a adv_curr_image for several steps
            if while_number <= 5:
                adv_curr_image = curr_image.clone().detach()
            else:
                adv_curr_image = repairset.inputs[fgsm_ith-1].clone().unsqueeze(0).detach()
            pgd_ith+=1
            logging.info(f'--PGD sample {pgd_ith} image')
            for i in range(10):
                adv_curr_image = adv_curr_image + torch.empty_like(adv_curr_image).uniform_(-radius, radius).detach()
                # adv_curr_image = torch.clamp(adv_curr_image, min=0, max=1).detach()
                step = 0
                while True:
                    
                    adv_curr_image.requires_grad = True
                    outputs = net(adv_curr_image)
                    cost = loss(outputs, label)
                    grad = torch.autograd.grad(cost, adv_curr_image, retain_graph=False, create_graph=False)[0]
                    
                    adv_curr_image = adv_curr_image.detach() + 2/255*grad.sign()
                    delta = torch.clamp(adv_curr_image - curr_image, min=-radius, max=radius)
                    adv_curr_image = (curr_image + delta).detach()

                    outputs = net(adv_curr_image)
                    _, predicted = torch.max(outputs.data, 1)
                    if label != predicted:
                        attack_datas.append(adv_curr_image)
                        num+=1
                    else:
                        pass
                    
                    
                    if step <= 50:
                        step+=1
                        continue
                
                    else:
                        curr_adv_images_cat = torch.cat(attack_datas)
                        break

            full_adv_images_cat.append(curr_adv_images_cat)
            data_number_record_list.append(curr_adv_images_cat.size(0))

        # full_adv_images_cat = torch.cat(full_adv_images_cat)
        return full_adv_images_cat, data_number_record_list
    # with torch.no_grad():

    logging.info(f'get the feature of the repairset, testset and trainset')
    feature_extractor, classifier = net.split()
    feature_extractor.eval()
    feature_extractor.to(device)
    
    with torch.no_grad():
        feature_repairset_input = feature_extractor(repairset.inputs)
        feature_testset_input = feature_extractor(testset.inputs)
        feature_trainset_input = feature_extractor(trainset.inputs)
        feature_attack_testset_input = feature_extractor(attack_testset.inputs)
    
    # 查看显存占用并释放缓存
    torch.cuda.empty_cache()


    feature_repairset = CifarPoints(feature_repairset_input, repairset.labels)
    feature_testset = CifarPoints(feature_testset_input, testset.labels)
    feature_trainset = CifarPoints(feature_trainset_input, trainset.labels)
    feature_attack_testset = CifarPoints(feature_attack_testset_input, attack_testset.labels)
    
    def get_the_box_bounds_of_features(sample_feature_list, 
                                       feature_repairset,data_number_record_list):
        for i in range(len(sample_feature_list)):
            sample_features = sample_feature_list[i]

            # sample_features = feature_extractor(sample_datas)
            if i == 0:
                bound_mins = torch.zeros((len(data_number_record_list), *sample_features.shape[1:]), device = device)
                bound_maxs = torch.zeros((len(data_number_record_list), *sample_features.shape[1:]), device = device)

        # get the box bound of the features
        # the shape of the sample_features is (n_sample, *feature_dim)
        # bound_mins = torch.zeros((len(data_number_record_list), *sample_features.shape[1:]), device = device)
        # bound_maxs = torch.zeros((len(data_number_record_list), *sample_features.shape[1:]), device = device)
        # for i,data_number in enumerate(data_number_record_list):
        #     if i == 0:
            # bound_mins[i] = sample_features[:data_number].min(dim=0)[0]
            # bound_maxs[i] = sample_features[:data_number].max(dim=0)[0]
            bound_mins[i] = sample_features.min(dim=0)[0]
            bound_maxs[i] = sample_features.max(dim=0)[0]
            # else:
            # bound_mins[i] = sample_features[sum(data_number_record_list[:i]):sum(data_number_record_list[:i+1])].min(dim=0)[0]
            # bound_maxs[i] = sample_features[sum(data_number_record_list[:i]):sum(data_number_record_list[:i+1])].max(dim=0)[0]
        # loose the bound
        # bound_mins = torch.clamp(bound_mins - (args.loose_bound)*bound_mins, min=-1e-8)
        # bound_maxs = torch.clamp(bound_maxs + (args.loose_bound)*bound_maxs, min=0)
        bound_mins = bound_mins - (args.loose_bound)*bound_mins
        bound_maxs = bound_maxs + (args.loose_bound)*bound_maxs
                                 


        # union the every row of bound with the every element of repairset
        # the shape of the bound_mins and bound_maxs is (repairset.shape[0], *feature_dim)
        # the shape of the feature_repairset is (repairset.shape[0], *feature_dim)
        bound_mins_union = torch.cat((bound_mins.unsqueeze(-1), feature_repairset.inputs.unsqueeze(-1)), dim=-1).min(dim=-1)[0]
        bound_maxs_union = torch.cat((bound_maxs.unsqueeze(-1), feature_repairset.inputs.unsqueeze(-1)), dim=-1).max(dim=-1)[0]

        return bound_mins_union, bound_maxs_union
    
    
    # sample_features_list = []
    feature_lb_list = []
    feature_ub_list = []
    logging.info(f'sample the datas from the radius of the original data with pgd attack')
    for o in range(args.divided_repair):

        part = o
        if o != args.divided_repair - 1:
            tempset = CifarPoints(originalset.inputs[int(part*len(originalset.inputs)/args.divided_repair):int((part+1)*len(originalset.inputs)/args.divided_repair)], originalset.labels[int(part*len(originalset.labels)/args.divided_repair):int((part+1)*len(originalset.labels)/args.divided_repair)]) 
        else:
            tempset = CifarPoints(originalset.inputs[int(part*len(originalset.inputs)/args.divided_repair):], originalset.labels[int(part*len(originalset.labels)/args.divided_repair):])
        logging.info(f'--sample the {o} part of the originalset')
        sample_datas_list, data_number_record_list = sample_datas_from_radius(net, tempset, radius = args.repair_radius/255, datanumber = args.sample_amount)


        # get the features of every sample_datas_list
        sample_features_list = []
        with torch.no_grad():
            for i in range(len(sample_datas_list)):
                sample_features_list.append(feature_extractor(sample_datas_list[i]))
    
        # declare the sample_datas_list
        # sample_datas_list = None
            del sample_datas_list
            torch.cuda.empty_cache()

    # get the k-means center of every sample_datas_list
    # the shape of the center is (len(data_number_record_list), *feature_dim)
    # from sklearn.cluster import KMeans
    # import numpy as np
    # # A_flattened = [tensor.flatten().numpy() for tensor in A]
    # # A_np = np.array(A_flattened)
    # center_list = []
    # pca_reduction_dim = 3
    # for i in range(len(sample_features_list)):
    #     _,_,V = torch.pca_lowrank(sample_features_list[i], q=pca_reduction_dim)
    #     # get the demensionality reducted data
    #     reduction_data = torch.matmul(sample_features_list[i], V[:, :pca_reduction_dim])
    #     # get the center of every element of sample_datas
    #     # convert the tensor to numpy
    #     reduction_data_np = reduction_data.detach().cpu().numpy()
    #     kmeans = KMeans(n_clusters=1, random_state=0).fit(reduction_data_np)
    #     center_i = kmeans.cluster_centers_
    #     center_list.append(center_i)




    # def get_the_box_bounds_of_features(feature_extractor, sample_datas, batch_number):
    #     sample_features = feature_extractor(sample_datas)
    #     # get the box bound of the features
    #     # the shape of the sample_features is (n_sample, *feature_dim)
    #     # the shape of the bound_mins and bound_maxs is (n_sample/50, *feature_dim)
    #     bound_mins = torch.zeros((int(sample_features.shape[0]/batch_number), *sample_features.shape[1:]), device = device)
    #     bound_maxs = torch.zeros((int(sample_features.shape[0]/batch_number), *sample_features.shape[1:]), device = device)
        
    #     for i in range(args.repair_number):
    #         indexs = torch.arange(i, sample_features.shape[0], args.repair_number
    #                         , dtype=torch.long, device = device)
    #         bound_mins[i] = sample_features[indexs].min(dim=0)[0]
    #         bound_maxs[i] = sample_features[indexs].max(dim=0)[0]
    #     return bound_mins, bound_maxs







        

        



    # for every 50 samples, we need to get the box bound of the features

        with torch.no_grad():
            if o != args.divided_repair - 1:
                temp_feature_repairset = CifarPoints(feature_repairset.inputs[int(part*len(feature_repairset.inputs)/args.divided_repair):int((part+1)*len(feature_repairset.inputs)/args.divided_repair)], feature_repairset.labels[int(part*len(feature_repairset.labels)/args.divided_repair):int((part+1)*len(feature_repairset.labels)/args.divided_repair)])
            else:
                temp_feature_repairset = CifarPoints(feature_repairset.inputs[int(part*len(feature_repairset.inputs)/args.divided_repair):], feature_repairset.labels[int(part*len(feature_repairset.labels)/args.divided_repair):])
            feature_lb, feature_ub = get_the_box_bounds_of_features(sample_features_list, temp_feature_repairset, data_number_record_list)
        feature_lb_list.append(feature_lb)
        feature_ub_list.append(feature_ub)
        del sample_features_list
        torch.cuda.empty_cache()

        logging.info(f'--get the {o} part of the feature_lb and feature_ub')

    feature_lb = torch.cat(feature_lb_list)
    feature_ub = torch.cat(feature_ub_list)


    # n_repair = feature_lb.shape[0]
    feature_shape = feature_lb.shape[1:]
    # repairlist = [(data[0],data[1]) for data in zip(repairset.inputs, repairset.labels)]
    # repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, feature_shape= feature_shape,radius= args.repair_radius)
    feature_repair_feature_bounds = [(data[0],data[1]) for data in zip(feature_lb, feature_ub)]
    feature_repair_prop_list = CifarProp.all_feature_props(args.dom, bounds = feature_repair_feature_bounds, labels= repairset.labels, feature_shape = feature_shape)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    feature_all_props = AndProp(props=feature_repair_prop_list)
    feature_v = Bisecter(args.dom, feature_all_props)


    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins, batch_abs_bitmap)
        return feature_all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)
    


    feature_lb, feature_ub = feature_all_props.lbub(device)
    feature_bitmap = feature_all_props.bitmap(device)
    
    # get the bitmap of features of every dataset
    # feature_test_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_testset.inputs)
    # feature_repair_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_repairset.inputs)
    # feature_train_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_trainset.inputs)
    # feature_attack_test_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_attack_testset.inputs)

    # del sample_features_list
    torch.cuda.empty_cache()

    # with torch.no_grad():
    #     outs = net(repairset.inputs)
    #     predicted = outs.argmax(dim=1)

    # 
    input_shape = trainset.inputs.shape[1:]
    repairlist = [(data[0],data[1]) for data in zip(originalset.inputs, originalset.labels)]
    repair_prop_list = CifarProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius/255)
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(args.dom, all_props)


    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    test_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, testset.inputs)
    repairset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, repairset.inputs)
    # trainset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, trainset.inputs)
    attack_testset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, attack_testset.inputs)

    # test_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_testset.inputs, feature_testset.labels, repairset.labels, predicted)
    # trainset_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_trainset.inputs, feature_trainset.labels, repairset.labels, predicted)
    # repairset_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_repairset.inputs, feature_repairset.labels, repairset.labels, predicted)
    # attack_testset_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, feature_attack_testset.inputs, feature_attack_testset.labels, repairset.labels, predicted)

    # overvide the bitmap with the feature bitmap
    # all_props = feature_all_props 
    # test_bitmap = feature_test_bitmap
    # trainset_bitmap = feature_train_bitmap
    # repairset_bitmap = feature_repair_bitmap
    # attack_testset_bitmap = feature_attack_test_bitmap




    # test init accuracy
    logging.info(f'--Test repair set accuracy {eval_test(net, repairset)}')

    patch_lists = []
    n_repair = all_props.labels.shape[1]
    for i in range(n_repair):
        patch_net = Cifar_feature_patch_model_big(dom=args.dom,
            name = f'feature patch network {i}',input_dimension=feature_shape[0])
        patch_net.to(device)
        patch_lists.append(patch_net)
    logging.info(f'--Patch network: {patch_net}')

    # the number of repair patch network,which is equal to the number of properties 
    repair_net =  Netsum(args.dom, target_net = classifier, patch_nets= patch_lists, device=device)


    start = timer()

    if args.no_abs or args.no_refine:
        repair_abs_lb, repair_abs_ub, repair_abs_bitmap = feature_lb, feature_ub, feature_bitmap
    # else:
    #     # refine it at the very beginning to save some steps in later epochs
    #     # and use the refined bounds as the initial bounds for support network training
        
    #     repair_abs_lb, repair_abs_ub, repair_abs_bitmap = feature_v.split(feature_lb, feature_ub, feature_bitmap, classifier, args.refine_top_k,
    #                                                             tiny_width=args.tiny_width,
    #                                                             stop_on_k_all=args.start_abs_cnt, for_patch=False) #,for_support=False


    # # train the patch and original network
    # opti = Adam(repair_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # # accoridng to the net model
    # if args.net == 'CNN_small':
    #     opti.param_groups[0]['params'] = opti.param_groups[0]['params'][6:]
    # elif args.net == 'FNN_big' or args.net == 'FNN_small':
    #     opti.param_groups[0]['params'] = opti.param_groups[0]['params'][14:]
    # # scheduler = args.scheduler_fn(opti)  # could be None


    if args.no_refine:
        def get_orinet_out(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor) -> Tensor:
            """ Return the safety distances over abstract domain. """
            batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
            batch_abs_outs = net(batch_abs_ins)
            # lb, ub = batch_abs_outs.lbub()
            return batch_abs_outs
        def get_patch_and_ori_out(patch_net, oriout, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
            """ Return the safety distances over abstract domain. """
            ori_clone = oriout.clone()
            batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
            batch_abs_outs = patch_net(batch_abs_ins, batch_abs_bitmap, ori_clone)
            # lb, ub = batch_abs_outs.lbub()
            return batch_abs_outs
        
        
    

    def get_curr_setmap(dataset, bitmap, part):
        # recostruct the dataset and bitmap
        if part != args.divided_repair - 1:
            curr_map = bitmap[int(part*bitmap.shape[0]/args.divided_repair):int((part+1)*bitmap.shape[0]/args.divided_repair)]
            curr_set = CifarPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):int((part+1)*len(dataset.inputs)/args.divided_repair)], dataset.labels[int(part*len(dataset.labels)/args.divided_repair):int((part+1)*len(dataset.labels)/args.divided_repair)])
        else:
            curr_map = bitmap[int(part*bitmap.shape[0]/args.divided_repair):]
            curr_set = CifarPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):], dataset.labels[int(part*len(dataset.inputs)/args.divided_repair):])
        return curr_set, curr_map
        
        
        

    for o in range(args.divided_repair):
        accuracies = []  # epoch 0: ratio
        repair_acc = []
        # train_acc = []
        attack_test_acc = []
        certified = False
        epoch = 0

        opti = Adam(repair_net.parameters(), lr=args.lr)
        # accoridng to the net model
        # if args.net == 'vgg19':
        #     opti.param_groups[0]['params'] = opti.param_groups[0]['params'][6:]
        # elif args.net == 'FNN_big' or args.net == 'FNN_small':
        #     # save opti.param_groups[0]['params'][14:] for the patch network
        #     opti.param_groups[0]['params'] = opti.param_groups[0]['params'][14:]
            # scheduler = args.scheduler_fn(opti)
            # param_copy = opti.param_groups[0]['params'].copy()
        opti.param_groups[0]['params'] = opti.param_groups[0]['params'][2:]

        divide_repair_number = int(args.repair_number/args.divided_repair)

        # get the abstract output from the original network
        if o != args.divided_repair - 1:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = repair_abs_lb[o*divide_repair_number:(o+1)*divide_repair_number], repair_abs_ub[o*divide_repair_number:(o+1)*divide_repair_number], repair_abs_bitmap[o*divide_repair_number:(o+1)*divide_repair_number]
        else:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = repair_abs_lb[o*divide_repair_number:], repair_abs_ub[o*divide_repair_number:], repair_abs_bitmap[o*divide_repair_number:]


        
         # reset the dataset and bitmap
        curr_repairset, curr_repairset_bitmap = get_curr_setmap(feature_repairset, repairset_bitmap, o)
        curr_attack_testset, curr_attack_testset_bitmap = get_curr_setmap(feature_attack_testset, attack_testset_bitmap, o)

        # with torch.no_grad():
        #     orinet_out = get_orinet_out(net, curr_abs_lb[o*len()], curr_abs_ub)

        logging.info(f'[{utils.time_since(start)}] Start repair part {o}: {o*divide_repair_number}')
        
        while True:
            # first, evaluate current model
            
            logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
            if not args.no_pts:
                logging.info(f'Loaded {curr_repairset.real_len()} points for repair.')
                logging.info(f'Loaded {curr_attack_testset.real_len()} points for attack test.')
                # logging.info(f'Loaded {trainset.real_len()} points for training.')

            if not args.no_abs:
                logging.info(f'Loaded {len(curr_abs_lb)} abstractions for training.')
                with torch.no_grad():
                    full_dists = run_abs(repair_net,curr_abs_lb, curr_abs_ub, curr_abs_bitmap)
                logging.info(f'min loss {full_dists.min()}, max loss {full_dists.max()}.')
                if full_dists.max() <= 0:
                    certified = True
                    logging.info(f'All {len(curr_abs_lb)} abstractions certified.')
                else:
                    _, worst_idx = full_dists.max(dim=0)
                    # logging.info(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}, rule: {curr_abs_bitmap[worst_idx]}.')
                    logging.info(f'Max loss at rule: {curr_abs_bitmap[worst_idx]}.')
            # test the repaired model which combines the feature extractor, classifier and the patch network
            # accuracies.append(eval_test(finally_net, testset))

           

            
            accuracies.append(eval_test(repair_net, feature_testset, bitmap = test_bitmap))
            
            # repair_acc.append(eval_test(repair_net, feature_repairset, bitmap = curr_repairset_bitmap))
            repair_acc.append(eval_test(repair_net, curr_repairset, bitmap = curr_repairset_bitmap))

            # train_acc.append(eval_test(repair_net, feature_trainset, bitmap = trainset_bitmap))

            # attack_test_acc.append(eval_test(repair_net, feature_attack_testset, bitmap = curr_attack_testset_bitmap))
            attack_test_acc.append(eval_test(repair_net, curr_attack_testset, bitmap = curr_attack_testset_bitmap))

            logging.info(f'Test set accuracy {accuracies[-1]}.')
            logging.info(f'repair set accuracy {repair_acc[-1]}.')
            # logging.info(f'train set accuracy {train_acc[-1]}.')
            logging.info(f'attacked test set accuracy {attack_test_acc[-1]}.')

            # check termination
            # if certified and epoch >= args.min_epochs:
            if len(repair_acc) >= 3:
                if (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0) or (repair_acc[-1] == repair_acc[-2] and attack_test_acc[-1] == attack_test_acc[-2] and repair_acc[-1] == repair_acc[-3] and attack_test_acc[-1] == attack_test_acc[-3]) or certified:
                # all safe and sufficiently trained
                    break
            elif certified or (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0):
                break

            if epoch >= args.max_epochs:
                break

            epoch += 1
            certified = False
            logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')



            if args.no_refine:
                absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)

            # dataset may have expanded, need to update claimed length to date
            # if not args.no_pts:
            #     feature_trainset.reset_claimed_len()
            # if not args.no_abs:
            #     absset.reset_claimed_len()
            # if (not args.no_pts) and (not args.no_abs):
            #     ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            #     # need to enumerate both
            #     max_claimed_len = max(feature_trainset.claimed_len, absset.claimed_len)
            #     feature_trainset.claimed_len = max_claimed_len
            #     absset.claimed_len = max_claimed_len

            # if not args.no_pts:
            #     conc_loader = data.DataLoader(feature_trainset, batch_size=args.batch_size, shuffle=True)
            #     nbatches = len(conc_loader)
            #     conc_loader = iter(conc_loader)

                abs_loader = data.DataLoader(absset, batch_size=args.repair_batch_size, shuffle=False)
                # nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
                # abs_loader = iter(abs_loader)



            # total_loss = 0.
            # with torch.enable_grad():
            # full_epoch = 200



                # for i in range(nbatches):
                #     opti.zero_grad()
                #     batch_loss = 0.
                #     if not args.no_pts:
                #         batch_inputs, batch_labels = next(conc_loader)
                #         batch_conc_bitmap = get_bitmap(feature_lb, feature_ub, feature_bitmap, batch_inputs)
                #         batch_outputs = repair_net(batch_inputs, batch_conc_bitmap)
                #         batch_loss += 10*args.accuracy_loss(batch_outputs, batch_labels)
                #     if not args.no_abs:
                #         batch_abs_lb, batch_abs_ub, batch_abs_bitmap = next(abs_loader)
                #         batch_dists = run_abs(repair_net,batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                        
                #         #这里又对batch的loss求了个均值，作为最后的safe_loss(下面的没看懂，好像类似于l1)
                #         safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                #         # total_loss += safe_loss.item()
                #         batch_loss += safe_loss
                #     logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')

        
                #     batch_loss.backward()
                #     opti.step()
            # else:
                with torch.no_grad():
                    orinet_out = get_orinet_out(classifier, curr_abs_lb, curr_abs_ub)
                if args.repair_number < 100:
                    batch_size = 50
                else:
                    batch_size = 100
                full_epoch = 1
                for i in range(full_epoch):
                    for batch_abs_lb, batch_abs_ub, batch_abs_bitmap in abs_loader:
                        opti.zero_grad()
                        batch_loss = 0.

                        batchout = get_patch_and_ori_out(repair_net, orinet_out, batch_abs_lb, batch_abs_ub, batch_abs_bitmap)                
                        batch_dists = all_props.safe_dist(batchout, batch_abs_bitmap)
                    # time_end1=time.time()
                    # print('time cost1',time_end1-time_start1,'s')
                    # print(pp_cuda_mem(stamp='after1'))
                    # else:
                    # time_start2=time.time()
                    # batch_dists2 = run_abs(repair_net,batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                    # time_end2=time.time()
                    # print('time cost2',time_end2-time_start2,'s')
                    # print(pp_cuda_mem(stamp='after2'))

                    #这里又对batch的loss求了个均值，作为最后的safe_loss(下面的没看懂，好像类似于l1)
                        safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                        batch_loss += safe_loss
                    
                        logging.debug(f'Epoch {epoch}: {i :.2f}%. Batch loss {batch_loss.item()}')


                        
                        batch_loss.backward()
                        opti.step()
            # total_loss /= nbatches

            # # 修改学习率
            # if scheduler is not None:
            #     scheduler.step(total_loss)
            # logging.info(f'[{utils.time_since(start)}] At epoch {epoch}: avg accuracy training loss {total_loss}.')

            # Refine abstractions, note that restart from scratch may output much fewer abstractions thus imprecise.
            # TODO 这里继承在net上refine的输入，
            if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
                curr_abs_lb, curr_abs_ub, curr_abs_bitmap = feature_v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, repair_net,
                                                                    args.refine_top_k,
                                                                    # tiny_width=args.tiny_width,
                                                                    stop_on_k_new=args.refine_top_k,for_feature=True)
            pass
        train_time = timer() - start
        torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / f'Cifar-{args.net}-{args.repair_location}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}.pt'))
        logging.info(f'Accuracy at every epoch: {accuracies}')
        logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                    f'eventually the trained network got certified? {certified}, ' +
                    f'with {accuracies[-1]:.4f} accuracy on test set,' +
                    f'with {repair_acc[-1]:.4f} accuracy on repair set,' +
                    # f'with {train_acc[-1]:.4f} accuracy on train set,' +
                    f'with {attack_test_acc[-1]:.4f} accuracy on attack test set.')
        # torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / '2_9.pt'))
        # net.save_nnet(f'./ART_{nid.x.numpy().tolist()}_{nid.y.numpy().tolist()}_repair2p_noclamp_epoch_{epoch}.nnet',
        #             mins = bound_mins, maxs = bound_maxs) 
    
    # final test
    logging.info('final test')
    # logging.info(f'--Test set accuracy {eval_test(repair_net, testset, bitmap = test_bitmap)}')
    # logging.info(f'--Test repair set accuracy {eval_test(repair_net, repairset, bitmap = repairset_bitmap)}')
    # logging.info(f'--Test train set accuracy {eval_test(repair_net, trainset, bitmap = trainset_bitmap)}')
    # logging.info(f'--Test attack test set accuracy {eval_test(repair_net, attack_testset, bitmap = attack_testset_bitmap)}')
    logging.info(f'--Test set accuracy {eval_test(repair_net, feature_testset, bitmap = test_bitmap)}')
    logging.info(f'--Test repair set accuracy {eval_test(repair_net, feature_repairset, bitmap = repairset_bitmap)}')
    # logging.info(f'--Test train set accuracy {eval_test(repair_net, feature_trainset, bitmap = trainset_bitmap)}')
    logging.info(f'--Test attack test set accuracy {eval_test(repair_net, feature_attack_testset, bitmap = attack_testset_bitmap)}')
    logging.info(f'traing time {timer() - start}s')
    
        
    return epoch, train_time, certified, accuracies[-1]

def _run_repair(args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    res = []
    # for nid in nids:
    logging.info(f'For pgd attack net')
    outs = repair_cifar(args,weight_clamp=False)
    res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for pgd attack networks:')
    logging.info(avg_res)
    return




def test_goal_repair(parser: CifarArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        # 'max_abs_cnt': 
        'batch_size': 50,  # to make it faster

    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    # nids = acas.AcasNetID.goal_safety_ids(args.dom)
    if args.no_repair:
        print('why not repair?')
    else:
        _run_repair(args)
    return

def test(lr:float = 0.005, net:str = 'CNN_small',repair_radius:float = 0.1, repair_number = 200, refine_top_k = 300,
         train_datasize = 200, test_datasize = 2000, 
         accuracy_loss:str = 'CE'):
    test_defaults = {
        'net': net,
        'repair_location': 'feature',
        # 'patch_size': patch_size,
        'loose_bound': 0.0314, # 8/255: 1.5:45%, 1: 44%, 0.5:40% ,0.2:42.5%; 0.05:52%, 0.01: 52%; 
                            # 512: 0.01: 55%; 0.1, 53.5%; 0.5: 40.5%; 1: 26.5%;
        'sample_amount': 1000,
        'no_pts': True,
        'no_refine': True,
        'debug': False,
        'divided_repair': math.ceil(repair_number/100),
        'exp_fn': 'test_goal_repair',
        'refine_top_k': refine_top_k,
        'repair_batch_size': repair_number,
        'start_abs_cnt': 500,
        'max_abs_cnt': 1000,
        'no_repair': False,
        'repair_number': repair_number,
        'train_datasize':train_datasize,
        'test_datasize': test_datasize,
        'repair_radius': repair_radius,
        'lr': lr,
        'accuracy_loss': accuracy_loss,
        'tiny_width': repair_radius*0.0001,
        'min_epochs': 15,
        'max_epochs': 100,

        
    }
    parser = CifarArgParser(RES_DIR, description='CIFAR10 Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = globals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass


if __name__ == '__main__':

    # for lr in [0.005, 0.01]:
    #     for weight_decay in [0.0001, 0.00005]:
    #         # for k_coeff in [0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    #         for k_coeff in [0.4]:    
    #             for support_loss in ['SmoothL1', 'L2']:
    #                 for accuracy_loss in ['CE']:
    #                     # if lr == 0.005 and weight_decay == 0.0001 and k_coeff == 0.4 and support_loss == 'SmoothL1' and accuracy_loss == 'CE':
    #                     #     continue
    #                     # for repair_radius in [0.1, 0.05, 0.03, 0.01]:
    #                     test(lr=lr, weight_decay=weight_decay, k_coeff=k_coeff, repair_radius=0.1, support_loss=support_loss, accuracy_loss=accuracy_loss)
    # for radius in [0.1, 0.3]:
    


   # for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    for net in ['vgg19', 'resnet18']:
    # for net in ['resnet18']:
    # for net in ['vgg19']:
        # for patch_size in ['small', 'big']:
        # for patch_size in ['big']:
            # for radius in [4]: 
            for radius in [4,8]: 

            # for radius in [0.05,0.1,0.3]: #,0.1,0.3
                # for repair_number,test_number in zip([200],[2000]):
                # for repair_number,test_number in zip([50],[500]):
                # for repair_number,test_number in zip([1000],[10000]):
                for repair_number,test_number in zip([50,100,200,500,1000],[500,1000,2000,5000,10000]):
                    # if radius == 4 and (repair_number == 50 or repair_number == 100 or repair_number == 200):
                    #     continue
                    test(lr=10, net=net, repair_radius=radius, repair_number = repair_number, refine_top_k= 50, 
         train_datasize = 10000, test_datasize = test_number, 
         accuracy_loss='CE')
    # for radius in [0.05]: #,0.1,0.3
    #     # for repair_number,test_number in zip([500,1000],[5000,10000]):
    #     for repair_number,test_number in zip([500,1000],[5000,10000]):
    #         test(lr=0.01, net= 'CNN_small', repair_radius=radius, repair_number = repair_number, refine_top_k= 50, 
    #      train_datasize = 10000, test_datasize = test_number, 
    #      accuracy_loss='CE')

    # # get the features of the dataset using mnist_net.split
    # model1, model2 = net.split()
    # with torch.no_grad():
    #     feature_traindata = model1(trainset.inputs)
    # feature_trainset = CifarPoints(feature_traindata, trainset.labels)
    # with torch.no_grad():
    #     feature_testdata = model1(testset.inputs)
    #     feature_attack_testdata = model1(attack_testset.inputs)
    # feature_testset = CifarPoints(feature_testdata, testset.labels)
    # feature_attack_testset = CifarPoints(feature_attack_testdata, attack_testset.labels)