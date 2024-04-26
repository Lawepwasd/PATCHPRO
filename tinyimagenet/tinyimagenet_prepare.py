import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.models import resnet18, wide_resnet50_2, wide_resnet101_2, resnet50, resnet101, resnet152,vgg19
from tinyimagenet_utils import progress_bar
from pathlib import Path
from wide_resnet import Wide_Resnet_model_34_10
from tinyimagenet_utils import Wide_Resnet_101_2_model,Resnet_152_model
import sys
root = Path(__file__).resolve().parent.parent
py_file_location = root / 'PatchART'
sys.path.append(os.path.abspath(py_file_location))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from collections import OrderedDict
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
from TRADES.trades import trades_loss
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from torchattacks import AutoAttack
from common.repair_moudle import Netsum, NetFeatureSumPatch
from DiffAbs.DiffAbs import deeppoly
    # from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
from tinyimagenet_utils import TinyImagenet_feature_patch_model,TinyImagenetProp


def training(net_name:str, lr: float, resume:bool):
    '''
    is_adv: whether to use adversarial training by Trades
    '''
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    train_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(64, padding_mode='edge'),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))



    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    # prepare training
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)
    print('train_loader', len(trainloader))
    # print('==> Building model..')
    if net_name == 'vgg19':
        print('==> Building VGG model..')
        net = vgg19(pretrained=True)
        net.classifier[6] = nn.Linear(4096, 200)
        # net = VGG('VGG19')
    if net_name == 'resnet18':
        print('==> Building ResNet model..')
        net = resnet18(pretrained=True)
        net.fc = nn.Linear(512, 200)
    elif net_name == 'resnet50':
    # net = VGG('VGG19')
        print('==> Building ResNet model..')
        net = resnet50(num_classes=200)
    elif net_name == 'resnet101':
        print('==> Building ResNet model..')
        net = resnet101(num_classes=200)
    elif net_name == 'resnet152':
        print('==> Building ResNet model..')
        net = resnet152(pretrained=True)
        net.fc = nn.Linear(2048, 200)
        # net = resnet152(num_classes=200)
    elif net_name == 'wide_resnet34_10':
        print('==> Building Wide ResNet model..')
        net = Wide_Resnet_model_34_10()
    elif net_name == 'wide_resnet50_2':
        print('==> Building Wide ResNet model..')
        net = wide_resnet50_2(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    elif net_name == 'wide_resnet101_2':
        print('==> Building Wide ResNet model..')
        net = wide_resnet101_2(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    net = net.to(device)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('./checkpoint/ckpt.pth')
        # if net_name == 'vgg19':
        #     checkpoint = torch.load('./checkpoint/vgg19.pth')
        # elif net_name == 'resnet50':
        #     checkpoint = torch.load('./checkpoint/resnet50.pth')
        # elif net_name == 'wide_resnet34_10':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_34_10.pth')
        # elif net_name == 'resnet101':
        #     checkpoint = torch.load('./checkpoint/resnet101.pth')
        if net_name == 'resnet152':
            checkpoint = torch.load('./checkpoint_origin/resnet152.pth')
        elif net_name == 'wide_resnet101_2':
            checkpoint = torch.load('./checkpoint_origin/wide_resnet_101_2.pth')
        # elif net_name == 'wide_resnet50_2':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_50_2.pth')
        # net.load_state_dict(checkpoint['net'])
        if 'acc' in checkpoint.keys():
            state = checkpoint['net']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else :
            state = checkpoint
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        # torch.save(new_state_dict,f"./model/tiny_imagenet/resnet18.pth")
        net.load_state_dict(new_state_dict)
        


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)





    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    for epoch in range(start_epoch, start_epoch+200):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # train
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # if is_adv:
            #     loss = trades_loss(model=net,
            #                     x_natural=inputs,
            #                     y=targets,
            #                     optimizer=optimizer,
            #                     step_size=1/255,
            #                     epsilon=4/255,
            #                     perturb_steps=10,
            #                     beta=1.0)                
            # else:
            outputs = net(inputs)    
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                outputs = net(inputs) 
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        
        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Save checkpoint.
        # if not is_adv:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # if net_name == 'vgg19':
            #     torch.save(state, './checkpoint/vgg19.pth')
            # elif net_name == 'resnet18':
            #     torch.save(state, './checkpoint/resnet18.pth')
            # elif net_name == 'resnet50':
            #     torch.save(state, './checkpoint/resnet50.pth')
            # elif net_name == 'resnet101':
            #     torch.save(state, './checkpoint/resnet101.pth')
            if net_name == 'resnet152':
                torch.save(state, './checkpoint_origin/resnet152.pth')
            # elif net_name == 'wide_resnet34_10':
            #     torch.save(state, './checkpoint/wide_resnet_34_10.pth')
            elif net_name == 'wide_resnet101_2':
                torch.save(state, './checkpoint_origin/wide_resnet_101_2.pth')
            # elif net_name == 'wide_resnet50_2':
            #     torch.save(state, './checkpoint/wide_resnet_50_2.pth')
            best_acc = acc
        # step
        scheduler.step()

def adv_train_origin(net_name:str, radius: int, resume:bool, is_first,data_number = 500):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    origin_data,origin_label = torch.load(f'./data/tiny_imagenet/origin_data_{net_name}_{radius}.pt')
    origin_data = origin_data[:data_number]
    origin_label = origin_label[:data_number]
    origin_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(origin_data,origin_label), batch_size=32, shuffle=True)

    repair_data, repair_label = torch.load(f'./data/tiny_imagenet/train_attack_data_full_{net_name}_{radius}.pt',map_location=device)
    attack_data, attack_label = torch.load(f'./data/tiny_imagenet/test_attack_data_full_{net_name}_{radius}.pt',map_location=device)
    repair_data, repair_label = repair_data[:data_number],repair_label[:data_number]
    attack_data, attack_label = attack_data[:data_number*10],attack_label[:data_number*10]

    attack_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(attack_data,attack_label), batch_size=32, shuffle=True)
    
    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    

    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    print('train_loader', len(origin_data_loader))
    # print('==> Building model..')
    if net_name == 'resnet152':
        print('==> Building ResNet model..')
        net = resnet152(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    elif net_name == 'wide_resnet101_2':
        print('==> Building Wide ResNet model..')
        net = wide_resnet101_2(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    net = net.to(device)
    if is_first:
        if net_name == 'resnet152':
            checkpoint = torch.load(f'./checkpoint_origin/resnet152.pth')
        elif net_name == 'wide_resnet101_2':
            checkpoint = torch.load(f'./checkpoint_origin/wide_resnet_101_2.pth')
        if 'acc' in checkpoint.keys():
            state = checkpoint['net']
            new_state_dict = OrderedDict()
            for key, value in state.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(checkpoint)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('trades_train'), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('./checkpoint/ckpt.pth')
        # if net_name == 'vgg19':
        #     checkpoint = torch.load('./checkpoint/vgg19.pth')
        # elif net_name == 'resnet50':
        #     checkpoint = torch.load('./checkpoint/resnet50.pth')
        # elif net_name == 'wide_resnet34_10':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_34_10.pth')
        # elif net_name == 'resnet101':
        #     checkpoint = torch.load('./checkpoint/resnet101.pth')
        if net_name == 'resnet152':
            checkpoint = torch.load(f'./trades_train/resnet152_{radius}_{data_number}_origin_best.pth')
        elif net_name == 'wide_resnet101_2':
            checkpoint = torch.load(f'./trades_train/wide_resnet101_2_{radius}_{data_number}_origin_best.pth')
        # elif net_name == 'wide_resnet50_2':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_50_2.pth')
        # net.load_state_dict(checkpoint['net'])
        state = checkpoint['net']
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        # torch.save(new_state_dict,f"./model/tiny_imagenet/resnet18.pth")
        net.load_state_dict(new_state_dict)
        if not is_first:
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

    return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    print('radius',radius)
    import time
    start = time.time()
    time_sum = 0
    train_loss_best = 100000
    for epoch in range(start_epoch, 200):

        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_idx = 0
        for ori_data,ori_label in origin_data_loader:
            # train
            batch_idx+=1
            # inputs, targets = inputs.to(device), targets.to(device)
            ori_data,ori_label = ori_data.to(device), ori_label.to(device)
            optimizer.zero_grad()

            loss1 = trades_loss(model=net,
                                x_natural=ori_data,
                                y=ori_label,
                                optimizer=optimizer,
                                step_size=radius/(255*4),
                                epsilon=radius/255,
                                perturb_steps=10,
                                beta=1.0)                

            # outputs = net(inputs)
            # loss2 = criterion(outputs, targets)
            loss = loss1


            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                outputs = net(ori_data) 
                _, predicted = outputs.max(1)
                total += ori_label.size(0)
                correct += predicted.eq(ori_label).sum().item()

            progress_bar(batch_idx, len(origin_data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        mid = time.time()
        time_sum+=mid-start
        
        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Save checkpoint.
        
        acc = 100.*correct/total
        # if acc > best_acc:
        if train_loss < train_loss_best:
            print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'time':time_sum
            }
            if not os.path.isdir('trades_train'):
                os.mkdir('trades_train')
            torch.save(state, f'./trades_train/{net_name}_{radius}_{data_number}_origin_best.pth')
            train_loss_best = train_loss
    # elif net_name == 'wide_resnet50_2':
    #     torch.save(state, './checkpoint/wide_resnet_50_2.pth')
            # best_acc = acc
    # step
        start = mid
    scheduler.step()    

    best_checkpoint = torch.load(f'./trades_train/{net_name}_{radius}_{data_number}_origin_best.pth')
    state = best_checkpoint['net']
    new_state_dict = OrderedDict()
    for key, value in state.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)
    net.eval()
    # test
    rsr = 0
    with torch.no_grad():
        output = net(repair_data)
        rsr = (output.argmax(dim=1) == repair_label).sum().item()
        print(f"repair success rate {rsr/len(repair_data)}")
        # attack
        asr = 0
        for batch_idx, (inputs, targets) in enumerate(attack_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)
            asr += (output.argmax(dim=1) == targets).sum().item()
        print(f"attack success rate {asr/len(attack_data)}")
        # test
        tsr = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = net(inputs)
            tsr += (output.argmax(dim=1) == targets).sum().item()
        print(f"test success rate {tsr/len(test_set)}")

    with open(f'./tools/tinyimagenet/trade-tinyimagenet/result.txt','a') as f:
        f.write(f"For {net_name} {data_number} {radius} : repair:{rsr/len(repair_data)}, attack:{asr/len(attack_data)}, test:{tsr/len(test_set)}, time:{time_sum}\n")    

def adv_training(net_name:str, radius: int, resume:bool, is_first,data_number = 500):
    '''
    is_adv: whether to use adversarial training by Trades
    '''
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    origin_data,origin_label = torch.load(f'./data/tiny_imagenet/origin_data_{net_name}_{radius}.pt')
    origin_data = origin_data[:data_number]
    origin_label = origin_label[:data_number]
    train_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(64, padding_mode='edge'),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    # prepare training
    # repeat the origin data to the same size of train_set
    origin_data = origin_data.repeat(50000//data_number,1,1,1)
    origin_label = origin_label.repeat(50000//data_number)


    origin_data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(origin_data,origin_label), batch_size=128, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    print('train_loader', len(trainloader))
    # print('==> Building model..')
    if net_name == 'resnet152':
        print('==> Building ResNet model..')
        net = resnet152(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    elif net_name == 'wide_resnet101_2':
        print('==> Building Wide ResNet model..')
        net = wide_resnet101_2(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    net = net.to(device)
    if is_first:
        if net_name == 'resnet152':
            checkpoint = torch.load(f'./checkpoint_origin/resnet152.pth')
        elif net_name == 'wide_resnet101_2':
            checkpoint = torch.load(f'./checkpoint_origin/wide_resnet_101_2.pth')
        if 'acc' in checkpoint.keys():
            state = checkpoint['net']
            new_state_dict = OrderedDict()
            for key, value in state.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(checkpoint)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('trades_train'), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('./checkpoint/ckpt.pth')
        # if net_name == 'vgg19':
        #     checkpoint = torch.load('./checkpoint/vgg19.pth')
        # elif net_name == 'resnet50':
        #     checkpoint = torch.load('./checkpoint/resnet50.pth')
        # elif net_name == 'wide_resnet34_10':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_34_10.pth')
        # elif net_name == 'resnet101':
        #     checkpoint = torch.load('./checkpoint/resnet101.pth')
        if net_name == 'resnet152':
            checkpoint = torch.load(f'./trades_train/resnet152_{radius}_{data_number}.pth')
        elif net_name == 'wide_resnet101_2':
            checkpoint = torch.load(f'./trades_train/wide_resnet_101_2_{radius}_{data_number}.pth')
        # elif net_name == 'wide_resnet50_2':
        #     checkpoint = torch.load('./checkpoint/wide_resnet_50_2.pth')
        # net.load_state_dict(checkpoint['net'])
        state = checkpoint['net']
        new_state_dict = OrderedDict()
        for key, value in state.items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        # torch.save(new_state_dict,f"./model/tiny_imagenet/resnet18.pth")
        net.load_state_dict(new_state_dict)
        if not is_first:
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    print('radius',radius)
    for epoch in range(start_epoch, 50):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        # for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_idx = 0
        for (ori_data,ori_label),(inputs, targets) in zip(origin_data_loader,trainloader):
            # train
            batch_idx+=1
            inputs, targets = inputs.to(device), targets.to(device)
            ori_data,ori_label = ori_data.to(device), ori_label.to(device)
            optimizer.zero_grad()

            loss1 = trades_loss(model=net,
                                x_natural=ori_data,
                                y=ori_label,
                                optimizer=optimizer,
                                step_size=radius/(255*4),
                                epsilon=radius/255,
                                perturb_steps=10,
                                beta=1.0)                

            outputs = net(inputs)
            loss2 = criterion(outputs, targets)
            loss = loss1 + loss2


            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                outputs = net(inputs) 
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        
        # test
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Save checkpoint.
        
        acc = 100.*correct/total
        # if acc > best_acc:
        print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        }
        if not os.path.isdir('trades_train'):
            os.mkdir('trades_train')
        # if net_name == 'vgg19':
        #     torch.save(state, './checkpoint/vgg19.pth')
        # elif net_name == 'resnet18':
        #     torch.save(state, './checkpoint/resnet18.pth')
        # elif net_name == 'resnet50':
        #     torch.save(state, './checkpoint/resnet50.pth')
        # elif net_name == 'resnet101':
            # torch.save(state, './checkpoint/resnet101.pth')
        if net_name == 'resnet152':
            torch.save(state, f'./trades_train/resnet152_{radius}_{data_number}.pth')
        # elif net_name == 'wide_resnet34_10':
        #     torch.save(state, './checkpoint/wide_resnet_34_10.pth')
        elif net_name == 'wide_resnet101_2':
            torch.save(state, f'./trades_train/wide_resnet_101_2_{radius}_{data_number}.pth')
        # elif net_name == 'wide_resnet50_2':
        #     torch.save(state, './checkpoint/wide_resnet_50_2.pth')
        best_acc = acc
    # step
    scheduler.step()

def get_dataset():

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    train_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',
                                        transform=transforms.Compose([
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomCrop(64, padding_mode='edge'),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    trainset_inputs = []
    trainset_labels = []
    for i,data in enumerate(train_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        # convert labels from one-hot encoding 


        trainset_inputs.append(images)
        trainset_labels.append(labels)
        # print(f"batch {i} done")
    trainset_inputs = torch.cat(trainset_inputs)
    trainset_labels = torch.cat(trainset_labels)
    trainset_inputs_split = trainset_inputs#[:10000]
    trainset_labels_split = trainset_labels#[:10000]
    trainset_inputs_split.requires_grad = False
    trainset_labels_split.requires_grad = False
    torch.save((trainset_inputs_split, trainset_labels_split),'./data/tiny_imagenet/train.pt')
    # torch.save((trainset_inputs,trainset_labels),'./data/tiny_imagenet/train_norm00_full.pt')
    # test = datasets.tiny_imagenet('./data/', train=False,
    #                     transform=transform,
    #                     download=True)
    # test_loader = DataLoader(test, batch_size=128)

    testset_inputs = []
    testset_labels = []
    for i,data in enumerate(test_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        testset_inputs.append(images)
        testset_labels.append(labels)
        # print(f"batch {i} done")
    testset_inputs = torch.cat(testset_inputs)
    testset_labels = torch.cat(testset_labels)
    testset_inputs.requires_grad = False
    testset_labels.requires_grad = False
    torch.save((testset_inputs,testset_labels),'./data/tiny_imagenet/test.pt')



class PGD():
    def __init__(self,model,eps=0.3,alpha=3/255,steps=40,random_start=True):
        self.eps = eps
        self.model = model
        self.attack = "Projected Gradient Descent"
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default"]

    def forward(self,images,labels):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for step in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta).detach()

        return adv_images
    
    def forward_get_multi_datas(self,images,labels,number=5):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        attack_datas = []
        num = 0
        step = 0
        while True:
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            # is successful?
            outputs = self.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            if torch.all(labels != predicted):
                print(f"attack success {num}")
                attack_datas.append(adv_images)
                num+=1

            if step <= 40:
                step+=1
                print(f'process {step}')
                continue
            
            else:
                print(f"already collect {num} attacked data")
                # cat the attacked data
                if attack_datas == []:
                    return None
                adv_images_cat = torch.cat(attack_datas)
                # check every data is distinct
                adv_images_cat = torch.unique(adv_images_cat, dim=0)
                if adv_images_cat.size(0) >= number:
                    adv_images_cat = adv_images_cat[:number]
                    break
                elif adv_images_cat.shape[0] < number:
                    return None
                else:
                    print(f"{adv_images_cat.size(0)} datas are not enough, continue to attack")
                    continue

        # if attack_datas != []:
        return adv_images_cat
            
    


def pgd_get_data(net, radius = 2, multi_number = 10, data_num = 200, general = False):
    '''
    pgd attack to origin data in radius, then get the five distinct attacked data from one origin data
    '''
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if net == 'wide_resnet101_2':
        model = wide_resnet101_2(num_classes=200).to(device)
        state = torch.load(Path(root / 'checkpoint_origin/wide_resnet_101_2.pth'))
        # need replace the module.xx to xx
        if 'acc' in state.keys():
            # state = torch.load(Path(root / 'checkpoint_origin/wide_resnet_101_2.pth'))['net']
            state = state['net']
            new_state_dict = OrderedDict()
            for key, value in state.items():
                new_key = key.replace('module.', '') 
                new_state_dict[new_key] = value
            torch.save(new_state_dict,Path(root / 'checkpoint_origin/wide_resnet_101_2.pth'))
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)

    elif net == 'resnet152':
        model = resnet152(num_classes=200).to(device)
        # from resnet import ResNet18
        # model = ResNet18().to(device)
        # 'checkpoint_origin/wide_resnet_101_2.pth'
        # need replace the module.xx to xx
        # judge if state.keys contain 'acc'
        state = torch.load(Path(root / 'checkpoint_origin/resnet152.pth'))
        if 'acc' in state.keys():
            state = state['net']
            new_state_dict = OrderedDict()
            for key, value in state.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            torch.save(new_state_dict,Path(root / 'checkpoint_origin/resnet152.pth'))
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state)
    model.eval()

    # if general == True:
    #     train = torch.utils.data.Subset(train,range(40000,50000))
    
    train_data, train_label = torch.load(Path(root / 'data/tiny_imagenet/train.pt'))
    train_dataset = torch.utils.data.TensorDataset(train_data,train_label)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    origin_data = []
    origin_label = []
    pgd = PGD(model=model, eps=radius/255, alpha=radius/(255*4), steps=10, random_start=True)
    i = 0
    k = 0
    # from torchattacks import PGD
    collect_label = []
    for images,labels in train_loader:
        k+=1
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if labels != predicted:
                print(f"predict error {i}, continue")
                continue
        # pgd = PGD(model=model, eps=radius/255, alpha=2/255, steps=10, random_start=True)
        # adv_images = pgd(images,labels)
        # with torch.no_grad():
        #     outputs = model(adv_images)
        # _, predicted = torch.max(outputs.data, 1)
        # if labels != predicted:
        #     print(f"attack success {i}")
        adv_images = pgd.forward_get_multi_datas(images,labels,number=multi_number+1)
        # judge if the attack is successful
        if adv_images is None or adv_images.size(0) < multi_number+1:
            continue
        # judge if the label is already collected
        if i==0:
            print(f"initially collect data with label {labels}")
            collect_label.append(labels)
            i+=1
        else:
            if labels not in collect_label:
                print(f"collect data with label {labels}")
                collect_label.append(labels)
                i+=1
            else:
                print(f"already collect data with label {labels}, continue")
                if len(collect_label) == 200:
                    i+=1
                    pass
                else:
                    # if 200 classes are not collected and the label is already collected, continue
                    print(f"already collect {len(collect_label)} classes, continue")
                    continue


        origin_data.append(images)
        origin_label.append(labels)

        train_attacked_data.append(adv_images)
        multi_labels = labels.unsqueeze(1).expand(-1,multi_number+1)
        train_labels.append(multi_labels)
        if i >= data_num:
            break
    origin_data = torch.cat(origin_data)
    origin_label = torch.cat(origin_label).reshape(-1)
    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels).reshape(-1)
    # choose the first data from every multi_number+1 data,like 0,11,22,33,44,55,66,77,88,99 ...
    # then delete the train_repair_data from the train_attack_data
    train_repair_data = train_attack_data[multi_number::multi_number+1]
    train_repair_labels = train_attack_labels[multi_number::multi_number+1]
    data_mask = torch.ones_like(train_attack_data,dtype=torch.bool)
    data_mask[multi_number::multi_number+1] = False
    train_attack_data = train_attack_data[data_mask].reshape(-1,3,64,64)

    labels_mask = torch.ones_like(train_attack_labels,dtype=torch.bool)
    labels_mask[multi_number::multi_number+1] = False
    train_attack_labels = train_attack_labels[labels_mask]
    if general == True:
        torch.save((train_attack_data,train_attack_labels),f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')
        torch.save((origin_data,origin_label),f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
    else:
        torch.save((origin_data,origin_label),f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
        torch.save((train_attack_data,train_attack_labels),f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')
    with open(f'./data/tiny_imagenet/origin_data_{net}_{radius}.txt','a') as f:
        f.write(str(k))
        f.close()


def grad_none(net, radius):
    # load
    origin_data,origin_label = torch.load(f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
    train_attack_data,train_attack_labels = torch.load(f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')
    # grad none
    origin_data.requires_grad = False
    origin_label.requires_grad = False
    train_attack_data.requires_grad = False
    train_attack_labels.requires_grad = False
    test_attack_data.requires_grad = False
    test_attack_labels.requires_grad = False
    # save
    torch.save((origin_data,origin_label),f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
    torch.save((test_attack_data,test_attack_labels),f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')

def grad_none(net, radius):
    # load
    origin_data,origin_label = torch.load(f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
    train_attack_data,train_attack_labels = torch.load(f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')
    # grad none
    origin_data.requires_grad = False
    origin_label.requires_grad = False
    train_attack_data.requires_grad = False
    train_attack_labels.requires_grad = False
    test_attack_data.requires_grad = False
    test_attack_labels.requires_grad = False
    # save
    torch.save((origin_data,origin_label),f'./data/tiny_imagenet/origin_data_{net}_{radius}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/tiny_imagenet/train_attack_data_full_{net}_{radius}.pt')
    torch.save((test_attack_data,test_attack_labels),f'./data/tiny_imagenet/test_attack_data_full_{net}_{radius}.pt')

def test_robustness_acc(net_name:str,radius:int,is_test=True,repair_number=500):
    # load net
    state = torch.load(f'./trades_train/{net_name}_{radius}_{repair_number}.pth')
    if net_name == 'resnet152':
        print('==> Building ResNet model..')
        net = resnet152(pretrained=True)
        net.fc = nn.Linear(2048, 200)
        # net = resnet152(num_classes=200)
    elif net_name == 'wide_resnet101_2':
        print('==> Building Wide ResNet model..')
        net = wide_resnet101_2(pretrained=True)
        net.fc = nn.Linear(2048, 200)
    net = net.to(device)
    new_state_dict = OrderedDict()
    for key, value in state['net'].items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    net.load_state_dict(new_state_dict)
    # load data
    if is_test:
        test_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/val',
                                        transform=transforms.Compose([
                                            # transforms.CenterCrop(56),
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    else:
        test_data,test_label = torch.load(f'./data/tiny_imagenet/origin_data_{net_name}_{radius}.pt')
        test_data = test_data[:repair_number]
        test_label = test_label[:repair_number]
        test_set = torch.utils.data.TensorDataset(test_data,test_label)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    correct_sum = 0
    for images, labels in testloader:
        # count+=1
        images, labels = images.to(device), labels.to(device)
    # adv attack
        at = AutoAttack(model=net, norm='Linf', eps=radius/255, version='standard', verbose=False,
                        n_classes=200, steps=10)
        adv_images = at(images, labels)
        outs = net(adv_images)
        predicted = outs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        correct_sum += correct
        print(f'correct {correct}')
    print(f'correct_acc {correct_sum/len(testloader.dataset)}')
    with open(f'./data/tiny_imagenet/{net_name}_{radius}_robust_acc.txt','a') as f:
        f.write(f"robustness acc {correct_sum/len(testloader.dataset)}")
        f.close()




def load_trainset(data_num = 1000):
    train_set = torchvision.datasets.ImageFolder('./data/tiny-imagenet-200/train',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                        ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    train_data = []
    train_label = []
    for i,data in enumerate(train_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        train_data.append(images)
        train_label.append(labels)
        if i >= data_num:
            break
    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label)
    torch.save((train_data,train_label),'./data/tiny_imagenet/train_1000.pt')
# def read_care_model(net_name, data_number, radius, device):
#     net = torch.load(f'./tools/tinyimagenet/care/{net_name}-{data_number}-{radius}.pth')
#     # if net_name == 'resnet152':
#     #     print('==> Building ResNet model..')
#     #     net = resnet152(pretrained=True)
#     #     net.fc = nn.Linear(2048, 200)
#     #     # net = resnet152(num_classes=200)
#     # elif net_name == 'wide_resnet101_2':
#     #     print('==> Building Wide ResNet model..')
#     #     net = wide_resnet101_2(pretrained=True)
#     #     net.fc = nn.Linear(2048, 200)
#     # net = net.to(device)
#     # new_state_dict = OrderedDict()
#     # for key, value in state['net'].items():
#     #     new_key = key.replace('module.', '')
#     #     new_state_dict[new_key] = value

#     # net.load_state_dict(new_state_dict)

#     net.to(device)
#     return net.eval()
# def compare_with_care(net_name, data_number, radius, device):

#     model = read_care_model(net_name, data_number, radius, device)

    pass
def get_bitmap(in_lb, in_ub, in_bitmap, batch_inputs):
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
            # if in_lb.shape[0] > 50:
            is_in_list = []
            for i in range(batch_inputs_clone.shape[0]):
                batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                is_in_list.append(is_in_datai)
            is_in = torch.stack(is_in_list, dim=0)

        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device).to(torch.uint8)
        bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
        if bitmap_i.shape[0] != 0:
            bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
        else:
            pass
        return bitmap
def test_DSR(net_name, radius_bit, repair_number,device):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net

    radius= radius_bit/255
    state = torch.load(f'./trades_train/{net_name}_{radius_bit}_{repair_number}.pth')
    print(f'Using {device} device')
    if net_name == 'resnet152':
            print('==> Building ResNet model..')
            net = resnet152(pretrained=True)
            net.fc = nn.Linear(2048, 200)
            # net = resnet152(num_classes=200)
    elif net_name == 'wide_resnet101_2':
            print('==> Building Wide ResNet model..')
            net = wide_resnet101_2(pretrained=True)
            net.fc = nn.Linear(2048, 200)
    net = net.to(device)
    if 'acc' in state.keys():
        new_state_dict = OrderedDict()
        for key, value in state['net'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state)


    # if net_name == 'resnet152':
    #     frontier = Resnet_152_model(dom= deeppoly).to(device)
    #     frontier_state = torch.load(f"./model/tiny_imagenet/resnet152.pth")
    #     frontier.load_state_dict(frontier_state)
    # elif net_name == 'wide_resnet101_2':
    #     frontier = Wide_Resnet_101_2_model(dom= deeppoly).to(device)
    #     frontier_state = torch.load(f"./model/tiny_imagenet/wide_resnet101_2.pth")
    #     frontier.load_state_dict(frontier_state)
    # frontier.to(device)

    # frontier, rear  = frontier.split()

    # patch_lists = []
    # for i in range(repair_number):
    #     # if patch_format == 'small':
    #     patch_net = TinyImagenet_feature_patch_model(dom=deeppoly, name = f'feature patch network {i}', input_dimension=2048)
    #     # elif patch_format == 'big':
    #     #     patch_net = TinyImagenet_feature_patch_model(dom=deeppoly,name = f'big patch network {i}')
    #     patch_net.to(device)
    #     patch_lists.append(patch_net)
    # rear =  Netsum(deeppoly, target_net = rear, patch_nets= patch_lists, device=device)

    # rear_state = torch.load(f"./model/tiny_imagenet_patch_format/TinyImagenet-{net_name}-feature-repair_number{repair_number}-rapair_radius{radius_bit}.pt")
    # rear.load_state_dict(rear_state)
    # model2 = NetFeatureSumPatch(feature_sumnet=rear, feature_extractor=frontier)
    # if not os.path.isdir('model/tiny_imagenet_patch_format_sumformat'):
    #     os.mkdir('model/tiny_imagenet_patch_format_sumformat')
    # torch.save(model2.state_dict(),f"./model/tiny_imagenet_patch_format_sumformat/TinyImagenet-{net_name}-feature-repair_number{repair_number}-rapair_radius{radius_bit}-feature_sumnet.pt")
    # model2.eval()

    # model3 = read_care_model(net_name, repair_number, radius, device)



    # load data
    datas,labels = torch.load(f'./data/tiny_imagenet/origin_data_{net_name}_{radius_bit}.pt',map_location=device)
    # return
    
    datas = datas[:repair_number]
    labels = labels[:repair_number]

    # pgd
    # pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=50, random_start=True)
    # pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=50, random_start=True)
    from torchattacks import AutoAttack


    # attack
    # ori_step = 0
    # repair_step = 0
    # pgd_step = 0

    # get bitmap
    # from art.prop import AndProp
    # # from common.bisecter import Bisecter
    # repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    # repair_prop_list = TinyImagenetProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # # get the all props after join all l_0 ball feature property
    # # TODO squeeze the property list, which is the same as the number of label
    # all_props = AndProp(props=repair_prop_list)
    # in_lb, in_ub = all_props.lbub(device)
    # in_bitmap = all_props.bitmap(device)

    # bitmap = get_bitmap(in_lb, in_ub, in_bitmap, datas, device)

    p1 = 0
    p2 = 0
    # p3 = 0

    #load dataset
    dataset = torch.utils.data.TensorDataset(datas,labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


    for ith, (image, label) in enumerate(loader):
        image = image.to(device)
        label = label.to(device)

        print(f"advtrain {ith} batch")
        at1 = AutoAttack(net, norm='Linf', eps=radius, version='standard', verbose=False,n_classes=200,steps=10)
        adv_images1 = at1(image, label)
        correct1 = (net(adv_images1).argmax(dim=1) == label).sum().item()
        p1+=correct1
        print(f"correct1 {correct1}")

        # if net(adv_images1).argmax(dim=1)!= label:
        #     print("success1")
        #     p1 += 1
        # else:
        #     print("fail")

        # print(f"advrepair {ith} batch")
        # bitmap = get_bitmap(in_lb, in_ub, in_bitmap, image)
        # model2.feature_sumnet.bitmap = bitmap
        # at2 = AutoAttack(model2, norm='Linf', eps=radius, version='standard', verbose=False, bitmap=bitmap,n_classes=200,steps=10)
        # adv_images2 = at2(image, label)
        # correct2 = (model2(adv_images2).argmax(dim=1) == label).sum().item()
        # p2+=correct2
        # print(f"correct2 {correct2}")
        # at3 = AutoAttack(model3, norm='Linf', eps=radius, version='standard', verbose=False,n_classes=200,steps=10)
        # adv_images3 = at3(image, label)
        # if model3(adv_images3).argmax(dim=1) != label:
        #     print("success3")
        #     p3 += 1
        # else:
        #     print("fail")

            
    if not os.path.exists(f'./results/tiny_imagenet/repair/autoattack'):
        os.mkdir(f'./results/tiny_imagenet/repair/autoattack')
    # print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    with open(f'./results/tiny_imagenet/repair/autoattack/compare_DSR.txt','a') as f:
        f.write(f"For {net_name} {repair_number} {radius} : \\  adv-train:{p1/repair_number}, patch:{p2/repair_number} \\ \n")


def test_trades_RSR_RGR(net_name, radius_bit, repair_number,device):
    state = torch.load(f'./trades_train/{net_name}_{radius_bit}_{repair_number}.pth')
    # device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net_name == 'resnet152':
            print('==> Building ResNet model..')
            net = resnet152(pretrained=True)
            net.fc = nn.Linear(2048, 200)
            # net = resnet152(num_classes=200)
    elif net_name == 'wide_resnet101_2':
            print('==> Building Wide ResNet model..')
            net = wide_resnet101_2(pretrained=True)
            net.fc = nn.Linear(2048, 200)
    net = net.to(device)
    if 'acc' in state.keys():
        new_state_dict = OrderedDict()
        for key, value in state['net'].items():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(state)

    # load data
    RSR_datas,RSR_labels = torch.load(f'./data/tiny_imagenet/train_attack_data_full_{net_name}_{radius_bit}.pt',map_location=device
    )
    RGR_datas,RGR_labels = torch.load(f'./data/tiny_imagenet/test_attack_data_full_{net_name}_{radius_bit}.pt',map_location=device
    )
    RSR_datas = RSR_datas[:repair_number]
    RSR_labels = RSR_labels[:repair_number]
    RGR_datas = RGR_datas[:repair_number*10]
    RGR_labels = RGR_labels[:repair_number*10]
    # dataset
    RSR_dataset = torch.utils.data.TensorDataset(RSR_datas,RSR_labels)
    RGR_dataset = torch.utils.data.TensorDataset(RGR_datas,RGR_labels)
    RSR_loader = torch.utils.data.DataLoader(RSR_dataset, batch_size=32, shuffle=False)
    RGR_loader = torch.utils.data.DataLoader(RGR_dataset, batch_size=32, shuffle=False)
    # test
    net.eval()
    RSR_correct = 0
    RGR_correct = 0
    for images, labels in RSR_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        RSR_correct += predicted.eq(labels).sum().item()
    
    for images, labels in RGR_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        RGR_correct += predicted.eq(labels).sum().item()
    # mkdir
    if not os.path.exists(f'./results/tiny_imagenet/trades'):
        os.mkdir(f'./results/tiny_imagenet/trades')
    with open(f'./results/tiny_imagenet/trades/RSR_and_RGR.txt','a') as f:
        f.write(f"{net_name} {radius_bit} {repair_number} RSR:{RSR_correct/len(RSR_loader.dataset)} RGR:{RGR_correct/len(RGR_loader.dataset)}\n")
        f.close()



if __name__ == '__main__':
    # training('resnet18', 0.01, False)
    # training('vgg19', 0.01, True)
    # training('resnet50', 0.01, False)
    # training('wide_resnet50_2', 0.01, True)
    # training('wide_resnet101_2', 0.01, True)
    # adv_training('wide_resnet101_2', 2, True, False)
    # adv_training('resnet152', 2, False, True)
    # training('resnet152', 0.01, True)

    # training('wide_resnet34_10', 0.01, True)
    # training('vgg', 0.01, False)

    # get_dataset()
    # pgd_get_data('resnet152',radius=4,multi_number=10,data_num=1000,general=False)
    # pgd_get_data('wide_resnet101_2',radius=4,multi_number=10,data_num=1000,general=False)
    # pgd_get_data('resnet152',radius=2,multi_number=10,data_num=1000,general=False)
    # pgd_get_data('wide_resnet101_2',radius=2,multi_number=10,data_num=1000,general=False)
    
    # grad_none('resnet152',2)
    # grad_none('wide_resnet101_2',2)

    for radius in [2,4]:
        for data in [500,1000]:
            # for net_name in ['resnet152']:
            # for net_name in ['wide_resnet101_2']:

            for net_name in ['resnet152','wide_resnet101_2']:
                # adv_train_origin(net_name,radius,False,True, data_number=data)
                adv_train_origin(net_name,radius,True,False, data_number=data)

                # adv_training(net_name,radius,False,True,data_number=data)
    #             test_DSR(net_name,radius,data,device)
    # #             test_trades_RSR_RGR(net_name,radius,data,device)

    #             test_robustness_acc(net_name,radius,is_test=True)
    # load_trainset(1000)
                
    # compare_with_care('resnet152',1000,2,device)
    # test_trades_RSR_RGR('resnet152',2,1000,device)

    # cuda:0
    # adv_train_origin('resnet152',2,False,True, data_number=500)
    # adv_train_origin('resnet152',2,False,True, data_number=1000)
    # test_DSR('resnet152',2,500,device)
    # test_DSR('resnet152',2,1000,device)
    # test_robustness_acc('resnet152',2,is_test=True,repair_number=500)
    # test_robustness_acc('resnet152',2,is_test=True,repair_number=1000)
    
    
    # # cuda:1
    # adv_train_origin('resnet152',4,False,True, data_number=500)
    # adv_train_origin('resnet152',4,False,True, data_number=1000)
    # test_DSR('resnet152',4,500,device)
    # test_DSR('resnet152',4,1000,device)
    # test_robustness_acc('resnet152',4,is_test=True,repair_number=500)
    # test_robustness_acc('resnet152',4,is_test=True,repair_number=1000)

    # # cuda:2
    # adv_train_origin('wide_resnet101_2',2,False,True, data_number=500)
    # adv_train_origin('wide_resnet101_2',2,False,True, data_number=1000)
    # test_DSR('wide_resnet101_2',2,500,device)
    # test_DSR('wide_resnet101_2',2,1000,device)
    # test_robustness_acc('wide_resnet101_2',2,is_test=True,repair_number=500)
    # test_robustness_acc('wide_resnet101_2',2,is_test=True,repair_number=1000)

    # # cuda:3
    # adv_train_origin('wide_resnet101_2',4,False,True, data_number=500)
    # adv_train_origin('wide_resnet101_2',4,False,True, data_number=1000)
    # test_DSR('wide_resnet101_2',4,500,device)
    # test_DSR('wide_resnet101_2',4,1000,device)
    # test_robustness_acc('wide_resnet101_2',4,is_test=True,repair_number=500)
    # test_robustness_acc('wide_resnet101_2',4,is_test=True,repair_number=1000)