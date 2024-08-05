# ignore the warning
import warnings
warnings.filterwarnings("ignore")
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
from common.prop import AndProp
from common.bisecter import Bisecter
from common import exp, utils
from common.repair_moudle import Netsum, get_bitmap
from mnist.mnist_utils import MnistNet_CNN_small, MnistNet_FNN_big, MnistNet_FNN_small, MnistProp, Mnist_patch_model, Mnist_patch_model_small
# from mnist.u import MnistNet, MnistFeatureProp

device = torch.device(f'cuda:2')
MNIST_DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'MNIST' / 'processed'
MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'model' /'mnist'
# MNIST_NET_DIR = Path(__file__).resolve().parent.parent / 'pgd' /'model' 


RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'small' 
RES_DIR.mkdir(parents=True, exist_ok=True)


# RES_DIR_LABEL = Path(__file__).resolve().parent.parent / 'results' / 'mnist' / 'repair' / 'label'
# RES_DIR_LABEL.mkdir(parents=True, exist_ok=True)
# REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'patch_format'
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist_patch_format_small'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'mnist_label_format' 
LABEL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

class MnistArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)

        self.add_argument('--net', type=str, choices=['FNN_small', 'FNN_big', 'CNN_small'],
                          default='CNN_small', help='network architecture')

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        self.add_argument('--repair_number', type=int, default=50,
                          help='the number of repair datas')
        self.add_argument('--repair_batchsize', type=int, default=1,
                            help='the batchsize of repair datas')
        self.add_argument('--patch_size', type=str, default= 'big', 
                          choices=['big', 'small'], help='the size of patch network')
        self.add_argument('--label_repaired', type=bool, default=False,
                            help='if True, we use the label repaired method')


        self.add_argument('--repair_radius',type=float, default=0.2, 
                          help='the radius of repairing datas or features')

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

class MnistPoints(exp.ConcIns):
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
        #_attack_data_full
        suffix = 'train' if train else 'test'
        if train:
            fname = f'train_norm00.pt'  # note that it is using original data
            # fname = f'{suffix}_norm00.pt'
            # mnist_train_norm00_dir = "/pub/data/xxxx/"
            # combine = torch.load(mnist_train_norm00_dir+fname, device)
            combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
            inputs, labels = combine 
            inputs = inputs[:trainnumber]
            labels = labels[:trainnumber]
        else:
            if is_test_accuracy:
                fname = f'test_norm00.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                # inputs = inputs[:testnumber]
                # labels = labels[:testnumber]
            elif is_origin_data:
                fname = f'origin_data_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]
            
            elif is_attack_testset_repaired:
                fname = f'test_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:testnumber]
                labels = labels[:testnumber]
            elif is_attack_repaired:
                fname = f'train_attack_data_full_{net}_{radius}.pt'
                combine = torch.load(Path(MNIST_DATA_DIR, fname), device)
                inputs, labels = combine
                inputs = inputs[:repairnumber]
                labels = labels[:repairnumber]

            # clean_inputs, clean_labels = clean_combine
            # inputs = torch.cat((inputs[:testnumber], clean_inputs[:testnumber] ), dim=0)
            # labels = torch.cat((labels[:testnumber], clean_labels[:testnumber] ),  dim=0)
        
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass

def eval_test(net, testset: MnistPoints, bitmap: Tensor =None, categories=None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        if bitmap is None:
            outs = net(testset.inputs)
        else:
            outs = net(testset.inputs, bitmap)
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        # ratio = correct / len(testset)
        ratio = correct / len(testset.inputs)

        # per category
        if categories is not None:
            for cat in categories:
                idxs = testset.labels == cat
                cat_predicted = predicted[idxs]
                cat_labels = testset.labels[idxs]
                cat_correct = (cat_predicted == cat_labels).sum().item()
                cat_ratio = math.nan if len(cat_labels) == 0 else cat_correct / len(cat_labels)
                logging.debug(f'--For category {cat}, out of {len(cat_labels)} items, ratio {cat_ratio}')
    return ratio



def repair_mnist(args: Namespace)-> Tuple[int, float, bool, float]:
    fname = f'mnist_{args.net}.pth'
    fpath = Path(MNIST_NET_DIR, fname)

    # train number是修复多少个点
    # originalset = torch.load(Path(MNIST_DATA_DIR, f'origin_data_{args.repair_radius}.pt'), device)
    # originalset = MnistPoints(inputs=originalset[0], labels=originalset[1])
    originalset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_origin_data=True)
    repairset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, radius=args.repair_radius,is_attack_repaired=True)
    trainset = MnistPoints.load(train=True, device=device, net=args.net, repairnumber=args.repair_number, trainnumber=args.train_datasize)
    testset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize,is_test_accuracy=True)
    attack_testset = MnistPoints.load(train=False, device=device, net=args.net, repairnumber=args.repair_number, testnumber=args.test_datasize, radius=args.repair_radius,is_attack_testset_repaired=True)
    if args.net == 'CNN_small':
        net = MnistNet_CNN_small(dom=args.dom)
    elif args.net == 'FNN_big':
        net = MnistNet_FNN_big(dom=args.dom)
    elif args.net == 'FNN_small':
        net = MnistNet_FNN_small(dom=args.dom)
    net.to(device)
    net.load_state_dict(torch.load(fpath, map_location=device))

    logging.info(f'--Test repair set accuracy {eval_test(net, repairset)}')
    logging.info(f'--Test original set accuracy {eval_test(net, originalset)}')
    logging.info(f'--Test test set accuracy {eval_test(net, testset)}')
    logging.info(f'--Test attack test set accuracy {eval_test(net, attack_testset)}')
    logging.info(f'--Test train set accuracy {eval_test(net, trainset)}')

    # judge the batch_inputs is in which region of property


    # ABstraction part
    # set the box bound of the features, which the length of the box is 2*radius
    # bound_mins = feature - radius
    # bound_maxs = feature + radius

    # the steps are as follows:
    # 1.construct a mnist prop file include Mnistprop Module, which inherits from Oneprop
    # 2.use the features to construct the Mnistprop
    # 3.use the Mnistprop to construct the Andprop(join) 
    # n_repair = repairset.inputs.shape[0]
    input_shape = trainset.inputs.shape[1:]
    # repairlist = [(data[0],data[1]) for data in zip(repairset.inputs, repairset.labels)]
    # repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius)
    repairlist = [(data[0],data[1]) for data in zip(originalset.inputs, originalset.labels)]
    if args.label_repaired:
        repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius, tasktype='attack_input_label')
    else:
        repair_prop_list = MnistProp.all_props(args.dom, DataList=repairlist, input_shape= input_shape,radius= args.repair_radius)
    # get the all props after join all l_0 ball feature property
    all_props = AndProp(props=repair_prop_list)
    v = Bisecter(args.dom, all_props)


    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins, batch_abs_bitmap)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)
    


    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    # in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    # in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)
    torch.cuda.empty_cache()
    test_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, testset.inputs,device=device)
    repairset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, repairset.inputs,device=device)
    # trainset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, trainset.inputs)
    attack_testset_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, attack_testset.inputs,device=device)

    # test init accuracy
    # logging.info(f'--Test repair set accuracy {eval_test(net, repairset)}')
    # acc = eval_test(net, repairset, bitmap= test_bitmap)
    # repair part
    # the number of repair patch network,which is equal to the number of properties
    # n_repair = MnistProp.LABEL_NUMBER

    patch_lists = []
    # for i in range(args.repair_number):
    if args.label_repaired:
        n_repair = all_props.labels.shape[1]
    else:
        n_repair = args.repair_number
    for i in range(n_repair):
        if args.net == 'CNN_small':
            patch_net = Mnist_patch_model(dom=args.dom,
            name = f'{args.patch_size} patch network {i}').to(device)
        else:
            patch_net = Mnist_patch_model_small(dom=args.dom,
            name = f'{args.patch_size} patch network {i}').to(device)
        patch_lists.append(patch_net)
    logging.info(f'--{args.patch_size} patch network: {patch_net}')

    # the number of repair patch network,which is equal to the number of properties 
    repair_net =  Netsum(args.dom, target_net = net, patch_nets= patch_lists, device=device)


    start = timer()

    if args.no_abs or args.no_refine:
        repair_abs_lb, repair_abs_ub, repair_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        # refine it at the very beginning to save some steps in later epochs
        # and use the refined bounds as the initial bounds for support network training
        
        repair_abs_lb, repair_abs_ub, repair_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                                tiny_width=args.tiny_width,
                                                                stop_on_k_all=args.start_abs_cnt, for_patch=False) #,for_support=False


    # train the patch and original network
    # opti = Adam(repair_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # # accoridng to the net model
    # if args.net == 'CNN_small':
    #     opti.param_groups[0]['params'] = opti.param_groups[0]['params'][6:]
    # elif args.net == 'FNN_big' or args.net == 'FNN_small':
    #     # save opti.param_groups[0]['params'][14:] for the patch network
    #     param_copy = opti.param_groups[0]['params'].copy()
        # opti.param_groups[0]['params'] = opti.param_groups[0]['params'][14:]
    # scheduler = args.scheduler_fn(opti)  # could be None

    # TODO get the network output
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
            curr_set = MnistPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):int((part+1)*len(dataset.inputs)/args.divided_repair)], dataset.labels[int(part*len(dataset.labels)/args.divided_repair):int((part+1)*len(dataset.labels)/args.divided_repair)])
        else:
            curr_map = bitmap[int(part*bitmap.shape[0]/args.divided_repair):]
            curr_set = MnistPoints(dataset.inputs[int(part*len(dataset.inputs)/args.divided_repair):], dataset.labels[int(part*len(dataset.inputs)/args.divided_repair):])
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
        if args.net == 'CNN_small':
            opti.param_groups[0]['params'] = opti.param_groups[0]['params'][6:]
        elif args.net == 'FNN_big' or args.net == 'FNN_small':
            # save opti.param_groups[0]['params'][14:] for the patch network
            opti.param_groups[0]['params'] = opti.param_groups[0]['params'][14:]
            # scheduler = args.scheduler_fn(opti)
            # param_copy = opti.param_groups[0]['params'].copy()
        divide_repair_number = int(args.repair_number/args.divided_repair)
        
        # get the abstract output from the original network
        if o != args.divided_repair - 1:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = repair_abs_lb[o*divide_repair_number:(o+1)*divide_repair_number], repair_abs_ub[o*divide_repair_number:(o+1)*divide_repair_number], repair_abs_bitmap[o*divide_repair_number:(o+1)*divide_repair_number]
            # if args.net == 'FNN_big' or args.net == 'FNN_small':
            #     opti.param_groups[0]['params'] = param_copy[14+6*o*divide_repair_number:14+6*(o+1)*divide_repair_number]
        else:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = repair_abs_lb[o*divide_repair_number:], repair_abs_ub[o*divide_repair_number:], repair_abs_bitmap[o*divide_repair_number:]
            # if args.net == 'FNN_big' or args.net == 'FNN_small':
            #     opti.param_groups[0]['params'] = param_copy[14+6*o*divide_repair_number:]
        with torch.no_grad():
            orinet_out = get_orinet_out(net, curr_abs_lb, curr_abs_ub)
        
         # reset the dataset and bitmap
        curr_repairset, curr_repairset_bitmap = get_curr_setmap(repairset, repairset_bitmap, o)
        curr_attack_testset, curr_attack_testset_bitmap = get_curr_setmap(attack_testset, attack_testset_bitmap, o)

        # with torch.no_grad():
        #     orinet_out = get_orinet_out(net, curr_abs_lb[o*len()], curr_abs_ub)

        logging.info(f'[{utils.time_since(start)}] Start repair part {o}: {o*divide_repair_number}')
        while True:
            # first, evaluate current model
            
            logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
            if not args.no_pts:
                logging.info(f'Loaded {curr_repairset.real_len()} points for repair.')
                logging.info(f'Loaded {curr_attack_testset.real_len()} points for attack test.')
                logging.info(f'Loaded {trainset.real_len()} points for training.')

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

           

            
            accuracies.append(eval_test(repair_net, testset, bitmap = test_bitmap))
            
            repair_acc.append(eval_test(repair_net, curr_repairset, bitmap = curr_repairset_bitmap))

            # train_acc.append(eval_test(repair_net, trainset, bitmap = trainset_bitmap))

            attack_test_acc.append(eval_test(repair_net, curr_attack_testset, bitmap = curr_attack_testset_bitmap))

            logging.info(f'Test set accuracy {accuracies[-1]}.')
            logging.info(f'repair set accuracy {repair_acc[-1]}.')
            # logging.info(f'train set accuracy {train_acc[-1]}.')
            logging.info(f'attacked test set accuracy {attack_test_acc[-1]}.')

            # check termination
            # if certified and epoch >= args.min_epochs:
            if len(repair_acc) >= 3:
                if (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0) or (repair_acc[-1] == repair_acc[-2] and attack_test_acc[-1] == attack_test_acc[-2]) or certified:
                # all safe and sufficiently trained
                    break
            elif certified or (repair_acc[-1] == 1.0 and attack_test_acc[-1] == 1.0):
                break

            if epoch >= args.max_epochs:
                break

            epoch += 1
            certified = False
            logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')

            absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)

            # dataset may have expanded, need to update claimed length to date
            # if not args.no_pts:
            #     trainset.reset_claimed_len()
            # if not args.no_abs:
            #     absset.reset_claimed_len()
            # if (not args.no_pts) and (not args.no_abs):
            #     ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            #     # need to enumerate both
            #     max_claimed_len = max(trainset.claimed_len, absset.claimed_len)
            #     trainset.claimed_len = max_claimed_len
            #     absset.claimed_len = max_claimed_len

            # if not args.no_pts:
            #     conc_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            #     nbatches = len(conc_loader)
            #     conc_loader = iter(conc_loader)
            # if not args.no_abs:
                # if args.no_refine:
                # abs_loader = data.DataLoader(absset, batch_size=args.repair_batch_size, shuffle=False)
            # else:
            if args.repair_number < 100:
                batch_size = 50
            else:
                batch_size = 100
            abs_loader = data.DataLoader(absset, batch_size=batch_size, shuffle=False)
            # nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same           
            # abs_shuffled_order = list(abs_loader.sampler)
                
            # abs_shuffled_indices = [index for batch in abs_shuffled_order for index in batch]



            # total_loss = 0.
            # with torch.enable_grad():
            full_epoch = 1
            for i in range(full_epoch):
                for batch_abs_lb, batch_abs_ub, batch_abs_bitmap in abs_loader:

                    opti.zero_grad()

                    batch_loss = 0.
               
                    batchout = get_patch_and_ori_out(repair_net, orinet_out, batch_abs_lb, batch_abs_ub, batch_abs_bitmap)                
                    batch_dists = all_props.safe_dist(batchout, batch_abs_bitmap)
                    safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                    # total_loss += safe_loss.item()
                    batch_loss += safe_loss
                    
                    logging.debug(f'Epoch {epoch}: {i :.2f}%. Batch loss {batch_loss.item()}')

                    batch_loss.backward()
                    opti.step()


            
            if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
                curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, repair_net,
                                                                    args.refine_top_k,
                                                                    # tiny_width=args.tiny_width,
                                                                    stop_on_k_new=args.refine_top_k,for_feature=True)
            pass
        train_time = timer() - start
        if args.label_repaired:
            torch.save(repair_net.state_dict(), str(LABEL_MODEL_DIR / f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}-{args.patch_size}.pt'))
        else:
            torch.save(repair_net.state_dict(), str(REPAIR_MODEL_DIR / f'Mnist-{args.net}-repair_number{args.repair_number}-rapair_radius{args.repair_radius}-{args.patch_size}.pt'))
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
    logging.info(f'--Test set accuracy {eval_test(repair_net, testset, bitmap = test_bitmap)}')
    logging.info(f'--Test repair set accuracy {eval_test(repair_net, repairset, bitmap = repairset_bitmap)}')
    # logging.info(f'--Test train set accuracy {eval_test(repair_net, trainset, bitmap = trainset_bitmap)}')
    logging.info(f'--Test attack test set accuracy {eval_test(repair_net, attack_testset, bitmap = attack_testset_bitmap)}')
    logging.info(f'training time {timer() - start}s')
    
        
    return epoch, train_time, certified, accuracies[-1]

def _run_repair(args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    res = []
    # for nid in nids:
    logging.info(f'For pgd attack net')
    outs = repair_mnist(args)
    res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for pgd attack networks:')
    logging.info(avg_res)
    return




def test_goal_repair(parser: MnistArgParser):
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
         accuracy_loss:str = 'CE',patch_size = 'big',label_repaired = False):
    test_defaults = {
        'net': net,
        'patch_size': patch_size,
        'no_pts': False,
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
        'label_repaired': label_repaired,

        
    }
    # if label_repaired:
    #     parser = MnistArgParser(RES_DIR_LABEL, description='MNIST Correct by Construction, label repaired')
    # else:
    parser = MnistArgParser(RES_DIR, description='MNIST Correct by Construction')
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
    


    for net in ['FNN_small', 'FNN_big', 'CNN_small']:
    # for net in ['CNN_small']:
    # for net in ['FNN_small']:
        for patch_size in ['small']: 
        # for patch_size in ['small', 'big']:
        # for patch_size in ['big']:
        # for patch_size in ['big']:
            # for radius in [0.3]:
            # for radius in [0.05]:
            for radius in [0.05,0.1,0.3]: #,0.1,0.3
                # if net == 'FNN_small' and radius == 0.05:
                #     continue
                # for repair_number,test_number in zip([100],[1000]):
                # for repair_number,test_number in zip([50],[500]):
                for repair_number,test_number in zip([50,100,200,500,1000],[500,1000,2000,5000,10000]):
                # for repair_number,test_number in zip([1000],[10000]):
                # for repair_number,test_number in zip([200],[2000]):
                    test(lr=10, net=net, repair_radius=radius, repair_number = repair_number, refine_top_k= 50, 
         train_datasize = 10000, test_datasize = test_number, 
         accuracy_loss='CE',patch_size=patch_size,label_repaired=False)
