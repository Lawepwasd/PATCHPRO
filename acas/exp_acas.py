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
import acas_utils as acas


from common.prop import AndProp
from common.bisecter import Bisecter
from common import exp, utils
from common.repair_moudle import PatchNet, Netsum, get_bitmap

RES_DIR = Path(__file__).resolve().parent.parent / 'results' / 'acas' 
RES_DIR.mkdir(parents=True, exist_ok=True)
REPAIR_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model' / 'acasxu'
REPAIR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class AcasArgParser(exp.ExpArgParser):
    """ Parsing and storing all ACAS experiment configuration arguments. """

    def __init__(self, log_path: Optional[str], *args, **kwargs):
        super().__init__(log_path, *args, **kwargs)
        self.add_argument('--net', type=str, choices=[f'Acas_{i}_{j}'for j in range(1,10) for i in range(1,6)], default='1_1',
                          help='ACAS Xu network ID, e.g., 1_1')

        # use repair or nor
        self.add_argument('--no_repair', type=bool, default=True, 
                        help='not repair use incremental')
        

        # training
        self.add_argument('--accuracy_loss', type=str, choices=['L1', 'MSE', 'CE'], default='CE',
                          help='canonical loss function for concrete points training')
        # self.add_argument('--support_loss', type=str, choices=['L1', 'MSE', 'CE','BCE'], default='BCE',
        #                   help= 'canonical loss function for patch net training')
        self.add_argument('--sample_amount', type=int, default=5000,
                          help='specifically for data points sampling from spec')
        self.add_argument('--reset_params', type=literal_eval, default=False,
                          help='start with random weights or provided trained weights when available')

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
            ce = nn.CrossEntropyLoss()
            # *= -1 because ACAS picks smallest value as suggestion
            return ce(outs * -1., labels)

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


class AcasPoints(exp.ConcIns):
    """ Storing the concrete data points for one ACAS network sampled.
        Loads to CPU/GPU automatically.
    """
    @classmethod
    def load(cls, nid: acas.AcasNetID, train: bool, device):
        suffix = 'train' if train else 'test'
        fname = f'{str(nid)}orig_{suffix}.pt'  # note that it is using original data
        combine = torch.load(Path(acas.ACAS_DIR, fname), device)
        inputs, labels = combine
        assert len(inputs) == len(labels)
        return cls(inputs, labels)
    pass


def eval_test(net: acas.AcasNet, testset: AcasPoints, bitmap:Tensor,categories=None) -> float:
    """ Evaluate accuracy on test set.
    :param categories: e.g., acas.AcasOut
    """
    with torch.no_grad():
        outs = net(testset.inputs, bitmap) * -1
        outs.squeeze_(1)
        predicted = outs.argmax(dim=1)
        correct = (predicted == testset.labels).sum().item()
        ratio = correct / len(testset)

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



def repair_acas(nid: acas.AcasNetID, args: Namespace)-> Tuple[int, float, bool, float]:
    fpath = nid.fpath()
    net, bound_mins, bound_maxs = acas.AcasNet.load_nnet(fpath, args.dom, device)
    if args.reset_params:
        net.reset_parameters()
    logging.info(net)


    all_props = AndProp(nid.applicable_props(args.dom))
    v = Bisecter(args.dom, all_props)

    def run_abs(net, batch_abs_lb: Tensor, batch_abs_ub: Tensor, batch_abs_bitmap: Tensor) -> Tensor:
        """ Return the safety distances over abstract domain. """
        batch_abs_ins = args.dom.Ele.by_intvl(batch_abs_lb, batch_abs_ub)
        batch_abs_outs = net(batch_abs_ins, batch_abs_bitmap)
        return all_props.safe_dist(batch_abs_outs, batch_abs_bitmap)
    
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)
    in_lb = net.normalize_inputs(in_lb, bound_mins, bound_maxs)
    in_ub = net.normalize_inputs(in_ub, bound_mins, bound_maxs)
    n_repair = len(in_bitmap[0])
    input_size = 5
    hidden_size = [10,10,10]
    patch_lists = []
    for i in range(n_repair):
        patch_net = PatchNet(input_size=input_size, dom=args.dom, hidden_sizes=hidden_size, output_size=5,
            name = f'patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)



    # the number of repair patch network,which is equal to the number of properties 
    repair_net = Netsum(args.dom, target_net= net, patch_nets= patch_lists, device=device)


    # already moved to GPU if necessary
    trainset = AcasPoints.load(nid, train=True, device=device)
    testset = AcasPoints.load(nid, train=False, device=device)

    start = timer()

    if args.no_abs or args.no_refine:
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = in_lb, in_ub, in_bitmap
    else:
        
        curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(in_lb, in_ub, in_bitmap, net, args.refine_top_k,
                                                                tiny_width=args.tiny_width,
                                                                stop_on_k_all=args.start_abs_cnt,for_patch=False) #,for_support=False
    
    


    opti = Adam(repair_net.parameters(), lr=args.lr)
    opti.param_groups[0]['params'] = opti.param_groups[0]['params'][14:]
    scheduler = args.scheduler_fn(opti)  # could be None


    accuracies = []  # epoch 0: ratio
    certified = False
    epoch = 0

    while True:
        # first, evaluate current model
        logging.info(f'[{utils.time_since(start)}] After epoch {epoch}:')
        if not args.no_pts:
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
                logging.info(f'Max loss at LB: {curr_abs_lb[worst_idx]}, UB: {curr_abs_ub[worst_idx]}, rule: {curr_abs_bitmap[worst_idx]}.')

        #
        test_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, testset.inputs,device)
        accuracies.append(eval_test(repair_net, testset, bitmap = test_bitmap))
        logging.info(f'Test set accuracy {accuracies[-1]}.')

        # check termination
        if certified and epoch >= args.min_epochs:

            break

        if epoch >= args.max_epochs:
            break

        epoch += 1
        certified = False
        logging.info(f'\n[{utils.time_since(start)}] Starting epoch {epoch}:')

        absset = exp.AbsIns(curr_abs_lb, curr_abs_ub, curr_abs_bitmap)


        if not args.no_pts:
            trainset.reset_claimed_len()
        if not args.no_abs:
            absset.reset_claimed_len()
        if (not args.no_pts) and (not args.no_abs):
            ''' Might simplify this to just using the amount of abstractions, is it unnecessarily complicated? '''
            max_claimed_len = max(trainset.claimed_len, absset.claimed_len)
            trainset.claimed_len = max_claimed_len
            absset.claimed_len = max_claimed_len

        if not args.no_pts:
            conc_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(conc_loader)
            conc_loader = iter(conc_loader)
        if not args.no_abs:
            abs_loader = data.DataLoader(absset, batch_size=args.batch_size, shuffle=True)
            nbatches = len(abs_loader)  # doesn't matter rewriting len(conc_loader), they are the same
            abs_loader = iter(abs_loader)

        total_loss = 0.
        for i in range(nbatches):
            opti.zero_grad()
            batch_loss = 0.
            if not args.no_pts:
                batch_inputs, batch_labels = next(conc_loader)
                batch_conc_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, batch_inputs,device)
                batch_outputs = repair_net(batch_inputs, batch_conc_bitmap)
                batch_outputs.squeeze_(1)
                batch_loss += args.accuracy_loss(batch_outputs, batch_labels)
            if not args.no_abs:
                batch_abs_lb, batch_abs_ub, batch_abs_bitmap = next(abs_loader)
                batch_dists = run_abs(repair_net,batch_abs_lb, batch_abs_ub, batch_abs_bitmap)
                
                safe_loss = batch_dists.mean()  # L1, need to upgrade to batch_worsts to unlock loss other than L1
                total_loss += safe_loss.item()
                batch_loss += safe_loss
            logging.debug(f'Epoch {epoch}: {i / nbatches * 100 :.2f}%. Batch loss {batch_loss.item()}')
            batch_loss.backward()
            opti.step()


        
        total_loss /= nbatches

        if scheduler is not None:
            scheduler.step(total_loss)
        logging.info(f'[{utils.time_since(start)}] At epoch {epoch}: avg accuracy training loss {total_loss}.')

        if (not args.no_refine) and len(curr_abs_lb) < args.max_abs_cnt:
            curr_abs_lb, curr_abs_ub, curr_abs_bitmap = v.split(curr_abs_lb, curr_abs_ub, curr_abs_bitmap, repair_net,
                                                                args.refine_top_k,
                                                                stop_on_k_new=args.refine_top_k,for_patch=True)
        pass
    train_time = timer() - start
    logging.info(f'Accuracy at every epoch: {accuracies}')
    logging.info(f'After {epoch} epochs / {utils.pp_time(train_time)}, ' +
                 f'eventually the trained network got certified? {certified}, ' +
                 f'with {accuracies[-1]:.4f} accuracy on test set.')

    torch.save(repair_net, Path(REPAIR_MODEL_DIR, f'{str(nid)}repaired.pt')) 
    return epoch, train_time, certified, accuracies[-1]

def _run_repair(nids: List[acas.AcasNetID], args: Namespace):
    """ Run for different networks with specific configuration. """
    logging.info('===== start repair ======')
    res = []
    for nid in nids:
        logging.info(f'For {nid}')
        args.net = f'{nid.x}_{nid.y}'
        outs = repair_acas(nid, args)
        res.append(outs)

    avg_res = torch.tensor(res).mean(dim=0)
    logging.info(f'=== Avg <epochs, train_time, certified, accuracy> for {len(nids)} networks:')
    logging.info(avg_res)
    return




def test_goal_safety(parser: AcasArgParser):
    """ Q1: Show that we can train previously unsafe networks to safe. """
    defaults = {
        # 'start_abs_cnt': 5000,
        'batch_size': 100,  # to make it faster
        'min_epochs': 3,
        'max_epochs': 35
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    logging.info(utils.fmt_args(args))
    nids = acas.AcasNetID.goal_safety_ids(args.dom)
    # for nid_i in nids:
    if args.no_repair:
        logging.info('Why not repair?')
    else:
        _run_repair(nids, args)
    return

if __name__ == '__main__':
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    test_defaults = {
        'exp_fn': 'test_goal_safety',
        'batch_size': 100,  
        'min_epochs': 25,
        'max_epochs': 35,
        'lr': 0.001,
        'no_repair': False,
    }
    parser = AcasArgParser(RES_DIR, description='ACAS Xu Correct by Construction')
    parser.set_defaults(**test_defaults)
    args, _ = parser.parse_known_args()

    exp_fn = locals()[args.exp_fn]
    start = timer()
    exp_fn(parser)

    logging.info(f'Total Cost Time: {timer() - start}s.\n\n\n')
    pass
