

import sys
from pathlib import Path
from typing import List

import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))
root = Path(__file__).resolve().parent.parent
# sys.path.append(str(root))
from DiffAbs.DiffAbs import AbsDom, DeeppolyDom

from common.prop import AndProp



from acas.acas_utils import ACAS_DIR, AcasNetID, AcasNet, AcasOut, AcasProp
from common.utils import sample_points
def sift_data(dom: AbsDom, device = 'cuda'):
    for nid in AcasNetID.goal_safety_ids(dom=dom):
        fpath = nid.fpath()
        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        net = net.to(device)
        in_lbs = torch.tensor([bound_mins], device=device)
        in_ubs = torch.tensor([bound_maxs], device=device)

        in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
        in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)

        prop2 = AcasProp.property2(dom)
        lb, ub = prop2.lbub(device=device)
        lb = net.normalize_inputs(lb, bound_mins, bound_maxs)
        ub = net.normalize_inputs(ub, bound_mins, bound_maxs)

        data,label = torch.load(Path(ACAS_DIR, f'{str(nid)}orig_test.pt'),map_location=device)
        # judge the data is in [lb, ub]
        judge = ((data >= lb) & (data <= ub))
        judge = judge.all(dim=-1)
        # get the data which is in [lb, ub]
        data = data[~judge]
        label = label[~judge]
        # 
        torch.save((data,label),Path(ACAS_DIR, f'{str(nid)}orig_test.pt'))


def sample_original_data(dom: AbsDom, trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR,
                         device = 'cuda'):
    """ Sample the data from every trained network. Serve as training and test set.
    :param dom: the data preparation do not use abstraction domains, although the AcasNet constructor requires it.
    """
    for nid in AcasNetID.all_ids():
        fpath = nid.fpath()
        print('\rSampling for network', nid, 'picked nnet file:', fpath, end='')
        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        net = net.to(device)

        in_lbs = torch.tensor([bound_mins], device=device)
        in_ubs = torch.tensor([bound_maxs], device=device)

        in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
        in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)
        inputs = sample_points(in_lbs, in_ubs, K=trainsize+testsize)

        with torch.no_grad():
            outputs = net(inputs)
            labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

        # # it seems the prediction scores from original ACAS Xu network is very close
        # softmax = torch.nn.Softmax(dim=1)
        # loss = torch.nn.CrossEntropyLoss()
        # print(loss(softmax(outputs * -1), labels))

        train_inputs, test_inputs = inputs[:trainsize, ...], inputs[trainsize:, ...]
        train_labels, test_labels = labels[:trainsize, ...], labels[trainsize:, ...]

        torch.save((train_inputs, train_labels), Path(dir, f'{str(nid)}-orig-train.pt'))
        torch.save((test_inputs, test_labels), Path(dir, f'{str(nid)}-orig-test.pt'))
        print('\rSampled for network', nid, 'picked nnet file:', fpath)
    return


def sample_balanced_data(dom: AbsDom, trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR,
                         device = 'cuda'):
    assert trainsize % len(AcasOut) == 0 and testsize % len(AcasOut) == 0

    for nid in AcasNetID.balanced_ids():
        fpath = nid.fpath()
        print('Sampling for network', nid, 'picked nnet file:', fpath)
        net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
        net = net.to(device)

        in_lbs = torch.tensor([bound_mins], device=device)
        in_ubs = torch.tensor([bound_maxs], device=device)

        in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
        in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)

        res_inputs = [torch.tensor([]) for _ in range(len(AcasOut))]
        res_labels = [torch.tensor([]).long() for _ in range(len(AcasOut))]

        trainsize_cat = int(trainsize / len(AcasOut))
        testsize_cat = int(testsize / len(AcasOut))
        allsize_cat = trainsize_cat + testsize_cat
        while True:
            inputs = sample_points(in_lbs, in_ubs, K=trainsize+testsize)
            with torch.no_grad():
                outputs = net(inputs)
                labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

            all_filled = True
            for category in AcasOut:
                if len(res_inputs[category]) >= allsize_cat:
                    continue

                all_filled = False
                idxs = labels == category
                cat_inputs, cat_labels = inputs[idxs], labels[idxs]
                res_inputs[category] = torch.cat((res_inputs[category], cat_inputs), dim=0)
                res_labels[category] = torch.cat((res_labels[category], cat_labels), dim=0)

            if all_filled:
                break
            pass

        empty = torch.tensor([])
        train_inputs, train_labels = empty, empty.long()
        test_inputs, test_labels = empty, empty.long()

        for category in AcasOut:
            cat_inputs, cat_labels = res_inputs[category], res_labels[category]
            train_inputs = torch.cat((train_inputs, cat_inputs[:trainsize_cat, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:trainsize_cat, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[trainsize_cat:trainsize_cat+testsize_cat, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[trainsize_cat:trainsize_cat+testsize_cat, ...]), dim=0)
            pass

        # # it seems the prediction scores from original ACAS Xu network is very close
        # softmax = torch.nn.Softmax(dim=1)
        # loss = torch.nn.CrossEntropyLoss()
        # print(loss(softmax(outputs * -1), labels))

        with open(Path(dir, f'{str(nid)}-normed-train.pt'), 'wb') as f:
            torch.save((train_inputs, train_labels), f)
        with open(Path(dir, f'{str(nid)}-normed-test.pt'), 'wb') as f:
            torch.save((test_inputs, test_labels), f)
    return


def sample_balanced_data_for(dom: AbsDom, nid: AcasNetID, ignore_idxs: List[int],
                             trainsize: int = 10000, testsize: int = 5000, dir: str = ACAS_DIR,
                             device = 'cuda'):
    """ Some networks' original data is soooooo imbalanced.. Some categories are ignored. """
    assert len(ignore_idxs) != 0, 'Go to the other function.'
    assert all([0 <= i < len(AcasOut) for i in ignore_idxs])
    print('Sampling for', nid, 'ignoring output category', ignore_idxs)

    ncats = len(AcasOut) - len(ignore_idxs)
    train_percat = int(trainsize / ncats)
    test_percat = int(testsize / ncats)

    def trainsize_of(i: AcasOut):
        return 0 if i in ignore_idxs else train_percat

    def testsize_of(i: AcasOut):
        return 0 if i in ignore_idxs else test_percat

    fpath = nid.fpath()
    print('Sampling for network', nid, 'picked nnet file:', fpath)
    net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
    net = net.to(device)

    in_lbs = torch.tensor([bound_mins], device=device)
    in_ubs = torch.tensor([bound_maxs], device=device)

    in_lbs = net.normalize_inputs(in_lbs, bound_mins, bound_maxs)
    in_ubs = net.normalize_inputs(in_ubs, bound_mins, bound_maxs)

    res_inputs = [torch.tensor([]) for _ in range(len(AcasOut))]
    res_labels = [torch.tensor([]).long() for _ in range(len(AcasOut))]

    while True:
        inputs = sample_points(in_lbs, in_ubs, K=trainsize + testsize)
        with torch.no_grad():
            outputs = net(inputs)
            labels = (outputs * -1).argmax(dim=-1)  # because in ACAS Xu, minimum score is the prediction

        filled_cnt = 0
        for category in AcasOut:
            if len(res_inputs[category]) >= trainsize_of(category) + testsize_of(category):
                filled_cnt += 1

            if category not in ignore_idxs and len(res_inputs[category]) >= trainsize_of(category) + testsize_of(category):
                continue

            idxs = labels == category
            cat_inputs, cat_labels = inputs[idxs], labels[idxs]

            res_inputs[category] = torch.cat((res_inputs[category], cat_inputs), dim=0)
            res_labels[category] = torch.cat((res_labels[category], cat_labels), dim=0)
            pass

        if filled_cnt == len(AcasOut):
            break
        pass

    empty = torch.tensor([])
    train_inputs, train_labels = empty, empty.long()
    test_inputs, test_labels = empty, empty.long()

    for category in AcasOut:
        cat_inputs, cat_labels = res_inputs[category], res_labels[category]
        if category in ignore_idxs:
            amount = len(cat_inputs)
            pivot = int(amount * trainsize / (trainsize + testsize))
            train_inputs = torch.cat((train_inputs, cat_inputs[:pivot, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:pivot, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[pivot:, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[pivot:, ...]), dim=0)
        else:
            trainsize_cat = trainsize_of(category)
            testsize_cat = testsize_of(category)
            train_inputs = torch.cat((train_inputs, cat_inputs[:trainsize_cat, ...]), dim=0)
            train_labels = torch.cat((train_labels, cat_labels[:trainsize_cat, ...]), dim=0)
            test_inputs = torch.cat((test_inputs, cat_inputs[trainsize_cat:trainsize_cat + testsize_cat, ...]), dim=0)
            test_labels = torch.cat((test_labels, cat_labels[trainsize_cat:trainsize_cat + testsize_cat, ...]), dim=0)
        pass

    # # it seems the prediction scores from original ACAS Xu network is very close
    # softmax = torch.nn.Softmax(dim=1)
    # loss = torch.nn.CrossEntropyLoss()
    # print(loss(softmax(outputs * -1), labels))

    with open(Path(dir, f'{str(nid)}-normed-train.pt'), 'wb') as f:
        torch.save((train_inputs, train_labels), f)
    with open(Path(dir, f'{str(nid)}-normed-test.pt'), 'wb') as f:
        torch.save((test_inputs, test_labels), f)
    return


def inspect_data_for(dom: AbsDom, nid: AcasNetID, dir: str = ACAS_DIR, normed: bool = True,
                     device = 'cuda'):
    """ Inspect the sampled data from every trained network. To serve as training and test set. """
    fpath = nid.fpath()
    print('Loading sampled data for network', nid, 'picked nnet file:', fpath)
    props = AndProp(nid.applicable_props(dom))
    print('Shall satisfy', props.name)
    net, bound_mins, bound_maxs = AcasNet.load_nnet(fpath, dom)
    net = net.to(device)

    mid = 'normed' if normed else 'orig'
    train_inputs, train_labels = torch.load(Path(dir, f'{str(nid)}-{mid}-train.pt'), device)
    test_inputs, test_labels = torch.load(Path(dir, f'{str(nid)}-{mid}-test.pt'), device)

    assert len(train_inputs) == len(train_labels)
    assert len(test_inputs) == len(test_labels)
    print(f'Loaded {len(train_inputs)} training samples, {len(test_inputs)} test samples.')

    for category in AcasOut:
        cnt = (train_labels == category).sum().item() + (test_labels == category).sum().item()
        print(f'Category {category} has {cnt} samples.')
    print()

    with torch.no_grad():
        # because in ACAS Xu, minimum score is the prediction
        assert torch.equal(train_labels, (net(train_inputs) * -1).argmax(dim=-1))
        assert torch.equal(test_labels, (net(test_inputs) * -1).argmax(dim=-1))
    return

def test_p2(predict):
    predict_max_index = predict.argmax(dim=-1)
    # judge if the predict is in the range of [1, 4]
    judge = (predict_max_index >= 1) & (predict_max_index <= 4)
    acc = judge.sum().item()/len(judge)
    return acc

def test(net, device):
    from common.repair_moudle import PatchNet, Netsum
    from DiffAbs.DiffAbs import deeppoly
    # model_path = root / 'model' / 'acasxu' / f'AcasNetID_{net}_repaired.pt'
    repair_data_path = root / 'data' / 'acas' / 'acasxu_data_repair' / f'n{net[0]}{net[2]}_counterexample.pt'
    gene_data_path = root / 'data' / 'acas' / 'acasxu_data_gene' / f'n{net[0]}{net[2]}_counterexample_test.pt'
    # model = torch.load(model_path, map_location=device)
    repair_data = torch.load(repair_data_path, map_location=device)
    gene_data = torch.load(gene_data_path, map_location=device)
    # tranverse
    # gene_data = gene_data.transpose(1,0)
    repair_bitmap = [[1] for _ in range(len(repair_data))]
    gene_bitmap = [[1] for _ in range(len(gene_data))]
    repair_bitmap = torch.tensor(repair_bitmap, dtype=torch.int8, device=device)
    gene_bitmap = torch.tensor(gene_bitmap, dtype=torch.int8, device=device)
    # model = AcasNet()
    # repair_acc = test_p2(model(repair_data,repair_bitmap))
    # gene_acc = test_p2(model(gene_data,gene_bitmap))
    # print(f'net {net} repair_acc {repair_acc} gene_acc {gene_acc}')

if __name__ == '__main__':
    net_list = [f'{i}_{j}' for i in range(2, 6) for j in range(1,10) ]
    for net in net_list:
        test(net, device = 'cuda:2')

    # net_id = '2_2'
    # sift_data(DeeppolyDom(), device = 'cuda:2')