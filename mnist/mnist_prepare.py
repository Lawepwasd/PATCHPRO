import sys
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
# import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

py_file_location = "./pgd"
sys.path.append(os.path.abspath(py_file_location))
sys.path.append(str(Path(__file__).resolve().parent.parent))

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#set random seed
torch.manual_seed(1)
# print(f'Using {device} device')

# cuda prepare
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())
# print(torch.version.cuda)


class FNN_big_NeuralNet(nn.Module):
    # define the structure of the network
    def __init__(self):
        super(FNN_big_NeuralNet,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 200)

        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 32)

        self.fc7 = nn.Linear(32, 10)


        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x
class FNN_small_NeuralNet(nn.Module):
    # define the structure of the network
    def __init__(self):
        super(FNN_small_NeuralNet,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 50)

        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 32)

        self.fc7 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x

class CNN_small_NeuralNet(nn.Module):
    def __init__(self):
        super(CNN_small_NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

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
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    def forward_get_multi_datas(self,images,labels,number=5):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

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
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            # is successful?
            outputs = self.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            if torch.all(labels != predicted):
                print(f"attack success {num}")
                attack_datas.append(adv_images)
                num+=1

            if step <= 200:
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
            
    
    def forward_sumsteps(self, images, labels, device = None, bitmap = None):
        images = images.clone().detach()
        labels = labels.clone().detach()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        steps = 0
        acc_flag = 0
        for step in range(self.steps):
            adv_images.requires_grad = True
            if bitmap is not None:
                in_lb, in_ub, in_bitmap = bitmap
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images, device)
                outputs = self.model(adv_images, adv_bitmap)
            else:
                outputs = self.model(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            steps+=1
            # is successful?
            if bitmap is not None:
                adv_bitmap = get_bitmap(in_lb, in_ub, in_bitmap, adv_images, device)
                outputs = self.model(adv_images, adv_bitmap)
            else:
                outputs = self.model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            if labels != predicted:
                print(f"attack success {steps}")
                acc_flag = 1
                return steps, acc_flag
        
        if steps == self.steps:
            print(f"attack fail {steps}")

        return steps, acc_flag

import math
def train(net: str, device, epoch_num = 50):
    '''
    :param net: the name of the network, like "FNN_big", "FNN_small", "CNN_small"
    '''
    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=True)
    train_loader = DataLoader(train, batch_size=256)
    # iter_train = iter(train_loader)
    # train_nbatch = math.ceil(60000/128)
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    # model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epoch_num):
        epoch_loss = 0
        correct, total = 0,0
        for inputs,labels in train_loader:
        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred.indices == labels).sum().item()
        print("Epoch:",epoch+1, " Loss: ",epoch_loss," Accuracy:",correct/total)
        
    torch.save(model.state_dict(), f'./model/mnist_{net}.pth')
    return model


def test(net: str,device):
    # test
    test = datasets.MNIST('./data/', train=False,
                      transform=transforms.Compose([transforms.ToTensor(),]),
                      download=True)
    test_loader = DataLoader(test, batch_size=128)
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    # model = NeuralNet().to(device)
    model.load_state_dict(torch.load(f'./model/mnist_{net}.pth'))
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def pgd_attack(net: str,device):
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./model/mnist_{net}.pth"))
    model.eval()

    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=False)
    
    test = datasets.MNIST('./data/', train=False, transform=transforms.Compose([transforms.ToTensor(),]),download=False)

    train_loader = DataLoader(train, batch_size=256)
    # atk_images, atk_labels = iter_train.next()
    test_loader = DataLoader(test, batch_size=64)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    test_attacked_data = []
    test_labels = []
    test_attacked = []
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    pgd = PGD(model=model, eps=0.03, alpha=2/255, steps=10, random_start=True)
    i = 0
    for images,labels in train_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"train attack success {i}")
            train_attacked_data.append(adv_images)
            train_labels.append(labels)
            train_attacked.append(predicted)
        else:
            train_attacked_data.append(adv_images[labels != predicted])
            train_labels.append(labels[labels != predicted])
            train_attacked.append(predicted[labels != predicted])

    train_attack_data = torch.cat(train_attacked_data)
    train_attack_labels = torch.cat(train_labels)
    train_attacked = torch.cat(train_attacked)

    with torch.no_grad():
        outs = model(train_attack_data)
        predicted = outs.argmax(dim=1)
        correct = (predicted == train_attack_labels).sum().item()
        ratio = correct / len(train_attack_data)

    torch.save((train_attack_data,train_attack_labels),'./data/MNIST/processed/train_attack_data_full.pt')
    torch.save((train_attack_data[:5000],train_attack_labels[:5000]),'./data/MNIST/processed/train_attack_data_part_5000.pt')
    torch.save(train_attacked[:5000],'./data/MNIST/processed/train_attack_data_part_label_5000.pt')

    pgd = PGD(model=model, eps=0.05, alpha=1/255, steps=100, random_start=True)
    i = 0
    for images,labels in test_loader:
        i += 1
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward(images,labels)
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        if torch.all(labels != predicted):
            print(f"test attack success {i}")
            test_attacked_data.append(adv_images)
            test_labels.append(labels)
            test_attacked.append(predicted)
        else:
            test_attacked_data.append(adv_images[labels != predicted])
            test_labels.append(labels[labels != predicted])
            test_attacked.append(predicted[labels != predicted])
    test_attack_data = torch.cat(test_attacked_data)
    test_attack_labels = torch.cat(test_labels)
    test_attacked = torch.cat(test_attacked)

    torch.save((test_attack_data,test_attack_labels),'./data/MNIST/processed/test_attack_data_full.pt')
    torch.save((test_attack_data[:2500],test_attack_labels[:2500]),'./data/MNIST/processed/test_attack_data_part_2500.pt')
    torch.save(test_attacked[:2500],'./data/MNIST/processed/test_attack_data_part_label_2500.pt')


def stack():

    # 定义一个深层卷积神经网络
    class DeepCNN(nn.Module):
        def __init__(self):
            super(DeepCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(256 * 64 * 64, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(-1, 256 * 64 * 64)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    
    model = DeepCNN()

    
    input_data = torch.randn(320, 3, 128, 128).to('cuda:7')  # 8张128x128大小的彩色图片

    
    model.to('cuda:1')

    
    while(1):
        output = model(input_data)

        
        # print(output)

        
        # print(torch.cuda.max_memory_allocated() / 1e9, "GB")

def pgd_get_data(net, radius = 0.1, multi_number = 10, data_num = 200, general = False):
    '''
    pgd attack to origin data in radius, then get the five distinct attacked data from one origin data
    '''
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth"))
    model.eval()
    # pgd attack
    train = datasets.MNIST('./data/', train=True,
                       transform=transforms.Compose([transforms.ToTensor(),]),
                       download=False)
    if general == True:
        train = torch.utils.data.Subset(train,range(40000,50000))
    train_loader = DataLoader(train, batch_size=1)
    train_attacked_data = []
    train_labels = []
    train_attacked = []
    origin_data = []
    origin_label = []
    pgd = PGD(model=model, eps=radius, alpha=2/255, steps=10, random_start=True)
    i = 0
    k = 0
    for images,labels in train_loader:
        k+=1
        images = images.to(device)
        labels = labels.to(device)
        adv_images = pgd.forward_get_multi_datas(images,labels,number=multi_number+1)
        if adv_images is not None and adv_images.size(0) >= multi_number+1:
            i += 1
        else:
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
    train_attack_data = train_attack_data[data_mask].reshape(-1,1,28,28)

    labels_mask = torch.ones_like(train_attack_labels,dtype=torch.bool)
    labels_mask[multi_number::multi_number+1] = False
    train_attack_labels = train_attack_labels[labels_mask]
    if general == True:
        torch.save((train_attack_data,train_attack_labels),f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt')
        torch.save((origin_data,origin_label),f'./data/MNIST/processed/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt')
    else:
        torch.save((origin_data,origin_label),f'./data/MNIST/processed/origin_data_{net}_{radius}.pt')
        torch.save((train_repair_data,train_repair_labels),f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt')
        torch.save((train_attack_data,train_attack_labels),f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt')
    with open(f'./data/MNIST/processed/origin_data_{net}_{radius}.txt','a') as f:
        f.write(str(k))
        f.close()


def grad_none(net, radius):
    # load
    origin_data,origin_label = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt')
    train_attack_data,train_attack_labels = torch.load(f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt')
    test_attack_data,test_attack_labels = torch.load(f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt')
    # grad none
    origin_data.requires_grad = False
    origin_label.requires_grad = False
    train_attack_data.requires_grad = False
    train_attack_labels.requires_grad = False
    test_attack_data.requires_grad = False
    test_attack_labels.requires_grad = False
    # save
    torch.save((origin_data,origin_label),f'./data/MNIST/processed/origin_data_{net}_{radius}.pt')
    torch.save((train_attack_data,train_attack_labels),f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt')
    torch.save((test_attack_data,test_attack_labels),f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt')

def get_trainset_norm00():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    train = datasets.MNIST('./data/', train=True,
                        transform=transforms.Compose([transforms.ToTensor(),]),
                        download=True)
    train_loader = DataLoader(train, batch_size=128)
    trainset_inputs = []
    trainset_labels = []
    for i,data in enumerate(train_loader,0):
        # collect batch of data and labels, then save as a tuple
        images, labels = data
        trainset_inputs.append(images)
        trainset_labels.append(labels)
        # print(f"batch {i} done")
    trainset_inputs = torch.cat(trainset_inputs)
    trainset_labels = torch.cat(trainset_labels)
    trainset_inputs.requires_grad = False
    trainset_labels.requires_grad = False
    torch.save((trainset_inputs[:10000],trainset_labels[:10000]),'./data/MNIST/processed/train_norm00.pt')
    # torch.save((trainset_inputs,trainset_labels),'./data/MNIST/processed/train_norm00_full.pt')
    # 但它太大了，有180M

def adv_training(net, radius, data_num, device):
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth"))
    train_attack_data,train_attack_labels = torch.load(f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt',map_location=device)
    train_attack_data = train_attack_data[:data_num]
    train_attack_labels = train_attack_labels[:data_num]
    test_attack_data,test_attack_labels = torch.load(f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt',map_location=device)
    test_attack_data = test_attack_data[:data_num]
    test_attack_labels = test_attack_labels[:data_num]
    
    # dataset
    train_attack_dataset = torch.utils.data.TensorDataset(train_attack_data,train_attack_labels)
    test_attack_dataset = torch.utils.data.TensorDataset(test_attack_data,test_attack_labels)
    # data loader
    train_attack_loader = DataLoader(train_attack_dataset, batch_size=50)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=128)
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    model.train()
    for epoch in range(200):
        print(f"adv-training epoch {epoch}")
        # epoch_loss = 0
        # correct, total = 0,0
        for inputs,labels in train_attack_loader:

        # for i in range(train_nbatch):
        #     inputs,labels = iter_train.__next__()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return model.eval()

def adv_training_test_pgd(net, radius,device,epoch_n = 200, data_num = 10000):
    print(f'Using {device} device')
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth",map_location=device))

    # load data
    repair_data,repair_label = torch.load(f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt',map_location=device)
    attack_data,attack_label = torch.load(f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt',map_location=device)
    test_data, test_label = torch.load('./data/MNIST/processed/test_norm00.pt',map_location=device)
    origin_data,origin_label = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)

    origin_data,origin_label = origin_data[:data_num],origin_label[:data_num]
    repair_data,repair_label = repair_data[:data_num],repair_label[:data_num]
    attack_data,attack_label = attack_data[:data_num],attack_label[:data_num]

    # dataset
    origin_dataset = torch.utils.data.TensorDataset(origin_data,origin_label)
    
    # dataloader
    origin_loader = DataLoader(origin_dataset, batch_size=32)

    #train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    loss_sum_best = 100

    import time
    start = time.time()
    from torchattacks import PGD
    pgd_attack = PGD(model, eps=radius, alpha=radius/4, steps=10, random_start=True)
    
    for epoch in range(epoch_n):
        print(f"adv-training epoch {epoch}")
        loss_sum = 0
        for inputs,labels in origin_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()

            


            attack_in = pgd_attack(inputs, labels)
            outputs = model(attack_in)
            loss1 = criterion(outputs, labels)


            outputs = model(inputs)
            loss2 = criterion(outputs, labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if epoch % 1 == 0:
            print(f'Epoch [{epoch+1}/{200}], Loss: {loss_sum/len(origin_loader):.4f}')
            # save model
            # judge the loss is nan

            if loss_sum < loss_sum_best:
                torch.save(model.state_dict(),f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt")
                loss_sum_best = loss_sum
    model.load_state_dict(torch.load(f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt",map_location=device))
        
    # test
    model.eval()
    end = time.time()
    train_time = end - start
    # test
    rsr = 0
    with torch.no_grad():
        output = model(repair_data)
        rsr = (output.argmax(dim=1) == repair_label).sum().item()
        print(f"repair success rate {rsr/len(repair_data)}")
        # attack
        asr = 0
        output = model(attack_data)
        asr = (output.argmax(dim=1) == attack_label).sum().item()
        print(f"attack success rate {asr/len(attack_data)}")
        # test
        tsr = 0
        output = model(test_data)
        tsr = (output.argmax(dim=1) == test_label).sum().item()
        print(f"test success rate {tsr/len(test_data)}")
    with open(f'./tools/mnist/trade-mnist/result.txt','a') as f:
        f.write(f"For {net} {data_num} {radius} : repair:{rsr/len(repair_data)}, attack:{asr/len(attack_data)}, test:{tsr/len(test_data)}, time:{train_time}, epoch:{epoch_n} \n")

# judge the batch_inputs is in which region of property
from torch import Tensor
def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor, device):
    '''
    in_lb: n_prop * input
    in_ub: n_prop * input
    batch_inputs: batch * input
    '''
    with torch.no_grad():
    
        batch_inputs_clone = batch_inputs.clone().unsqueeze_(1)
        # distingush the photo and the property
        if len(in_lb.shape) == 2:
            batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1])
            is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
            is_in = is_in.all(dim=-1) # every input is in the region of property, batch * n_prop
        elif len(in_lb.shape) == 4:
            if in_lb.shape[0] > 600:
                is_in_list = []
                for i in range(batch_inputs_clone.shape[0]):
                    batch_inputs_compare_datai = batch_inputs_clone[i].clone().expand(in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                    is_in_datai = (batch_inputs_compare_datai >= in_lb) & (batch_inputs_compare_datai <= in_ub)
                    is_in_datai = is_in_datai.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
                    is_in_list.append(is_in_datai)
                is_in = torch.stack(is_in_list, dim=0)
            else:
                batch_inputs_clone = batch_inputs_clone.expand(batch_inputs.shape[0], in_lb.shape[0], in_lb.shape[1], in_lb.shape[2], in_lb.shape[3])
                is_in = (batch_inputs_clone >= in_lb) & (batch_inputs_clone <= in_ub)
                is_in = is_in.all(dim=(-1)).all(dim=(-1)).all(dim=(-1)) # every input is in the region of property, batch * n_prop
        # convert to bitmap
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]),device=device).to(torch.uint8)
        # is in is a batch * in_bitmap.shape[0] tensor, in_bitmap.shape[1] is the number of properties
        # the every row of is_in is the bitmap of the input which row of in_bitmap is allowed
        bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
        if bitmap_i.shape[0] != 0:
            bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
        else:
            pass

        return bitmap

def compare_pgd_step_length(net, patch_format, 
                            radius, repair_number):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net
    from common.repair_moudle import Netsum
    from DiffAbs.DiffAbs import deeppoly
    from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net == 'CNN_small':
        model1 = CNN_small_NeuralNet().to(device)
        orinet = MnistNet_CNN_small(dom=deeppoly)
    elif net == 'FNN_big':
        model1 = FNN_big_NeuralNet().to(device)
        orinet = MnistNet_FNN_big(dom=deeppoly)
    elif net == 'FNN_small':
        model1 = FNN_small_NeuralNet().to(device)
        orinet = MnistNet_FNN_small(dom=deeppoly)
    model1.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth"))

  



    orinet.to(device)
    patch_lists = []
    for i in range(repair_number):
        if patch_format == 'small':
            patch_net = Mnist_patch_model(dom=deeppoly, name = f'small patch network {i}')
        elif patch_format == 'big':
            patch_net = Mnist_patch_model(dom=deeppoly,name = f'big patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    model2 =  Netsum(deeppoly, target_net = orinet, patch_nets= patch_lists, device=device)
    model2.load_state_dict(torch.load(f"./model/patch_format/Mnist-{net}-repair_number{repair_number}-rapair_radius{radius}-{patch_format}.pt",map_location=device))

    model3 = adv_training(net,radius, data_num=repair_number, device=device)


    # load data
    datas,labels = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
    # return
    
    datas = datas[:repair_number]
    labels = labels[:repair_number]

    # pgd
    pgd1 = PGD(model=model1, eps=radius, alpha=2/255, steps=50, random_start=True)
    pgd2 = PGD(model=model2, eps=radius, alpha=2/255, steps=50, random_start=True)
    pgd3 = PGD(model=model3, eps=radius, alpha=2/255, steps=50, random_start=True)

    # attack
    ori_step = 0
    repair_step = 0
    pgd_step = 0

    # get bitmap
    from common.prop import AndProp
    from common.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    # v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    p1 = 0
    p2 = 0
    p3 = 0

    for image, label in zip(datas,labels):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        
        step1, ori_acc = pgd1.forward_sumsteps(image,label)
        step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
        step3, adt_acc = pgd3.forward_sumsteps(image,label)
        ori_step += step1
        repair_step += step2
        pgd_step += step3
        if ori_acc == 1:
            p1 += 1
        if repair_acc == 1:
            p2 += 1
        if adt_acc == 1:
            p3 += 1
            
    
    print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    with open(f'./data/MNIST/processed/compare_pgd_step_length.txt','a') as f:
        f.write(f"For {net} {radius} {data} {patch_format}: \\ ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3} \\ \n")

    # def test_two_dataset_has_no_same_data():
    #     # test the two dataset has no same data
    #     for i in range(datas.size(0)):
    #         for j in range(i+1,datas.size(0)):
    #             if torch.all(datas[i] == datas[j]):
    #                 print(f"same data {i} {j}")
    #                 return False
    #     return True

def compare_autoattack(net, patch_format, 
                            radius, repair_number):
    '''
    use the length of pgd steps to compare the hardness of attacking two model respectively
    the model1 is origin model, model2 is repaired model
    '''
    # load net
    from common.repair_moudle import Netsum
    from DiffAbs.DiffAbs import deeppoly
    from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net == 'CNN_small':
        model1 = CNN_small_NeuralNet().to(device)
        orinet = MnistNet_CNN_small(dom=deeppoly)
    elif net == 'FNN_big':
        model1 = FNN_big_NeuralNet().to(device)
        orinet = MnistNet_FNN_big(dom=deeppoly)
    elif net == 'FNN_small':
        model1 = FNN_small_NeuralNet().to(device)
        orinet = MnistNet_FNN_small(dom=deeppoly)
    model1.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth"))

  



    orinet.to(device)
    patch_lists = []
    for i in range(repair_number):
        if patch_format == 'small':
            patch_net = Mnist_patch_model(dom=deeppoly, name = f'small patch network {i}')
        elif patch_format == 'big':
            patch_net = Mnist_patch_model(dom=deeppoly,name = f'big patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    model2 =  Netsum(deeppoly, target_net = orinet, patch_nets= patch_lists, device=device)
    model2.load_state_dict(torch.load(f"./model/patch_format/Mnist-{net}-repair_number{repair_number}-rapair_radius{radius}-{patch_format}.pt",map_location=device))

    model3 = adv_training(net,radius, data_num=repair_number, device=device)


    # load data
    datas,labels = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
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
    from common.prop import AndProp
    from common.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    # v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    bitmap = get_bitmap(in_lb, in_ub, in_bitmap, datas, device)

    p1 = 0
    p2 = 0
    p3 = 0

    for ith, (image, label) in enumerate(zip(datas,labels)):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        at1 = AutoAttack(model1, norm='Linf', eps=radius, version='standard', verbose=True)
        adv_images1 = at1(image, label)
        if model1(adv_images1).argmax(dim=1)!= label:
            print("success1")
            p1 += 1
        else:
            print("fail")
        at2 = AutoAttack(model2, norm='Linf', eps=radius, version='standard', verbose=True, bitmap=bitmap)
        adv_images2 = at2(image, label)
        if model2(adv_images2, bitmap[ith]).argmax(dim=1) != label:
            print("success2")
            p2 += 1
        else:
            print("fail")
        at3 = AutoAttack(model3, norm='Linf', eps=radius, version='standard', verbose=True)
        adv_images3 = at3(image, label)
        if model3(adv_images3).argmax(dim=1) != label:
            print("success3")
            p3 += 1
        else:
            print("fail")
        
        # step1, ori_acc = pgd1.forward_sumsteps(image,label)
        # step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
        # step3, adt_acc = pgd3.forward_sumsteps(image,label)
        # ori_step += step1
        # repair_step += step2
        # pgd_step += step3
        # if ori_acc == 1:
        #     p1 += 1
        # if repair_acc == 1:
        #     p2 += 1
        # if adt_acc == 1:
        #     p3 += 1
            
    
    # print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    with open(f'./results/mnist/repair/autoattack/compare_autoattack_ac.txt','a') as f:
        f.write(f"For {net} {radius} {data} {patch_format}: \\  ori:{p1}, patch:{p2}, adv-train:{p3} \\ \n")

from TRADES.trades import trades_loss
def adv_training_test(net, radius,data_num, device, epoch_n=200):
    print(f'Using {device} device')
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./model/mnist/mnist_{net}.pth",map_location=device))

    # load data
    repair_data,repair_label = torch.load(f'./data/MNIST/processed/train_attack_data_full_{net}_{radius}.pt',map_location=device)
    attack_data,attack_label = torch.load(f'./data/MNIST/processed/test_attack_data_full_{net}_{radius}.pt',map_location=device)
    test_data, test_label = torch.load('./data/MNIST/processed/test_norm00.pt',map_location=device)
    origin_data,origin_label = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)

    origin_data,origin_label = origin_data[:data_num],origin_label[:data_num]
    repair_data,repair_label = repair_data[:data_num],repair_label[:data_num]
    attack_data,attack_label = attack_data[:data_num],attack_label[:data_num]

    # dataset
    origin_dataset = torch.utils.data.TensorDataset(origin_data,origin_label)
    
    # dataloader
    origin_loader = DataLoader(origin_dataset, batch_size=32)

    #train
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    loss_sum_best = 100
    import time
    start = time.time()
    for epoch in range(epoch_n):
        print(f"adv-training epoch {epoch}")
        loss_sum = 0
        for inputs,labels in origin_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            loss = trades_loss(model=model,
                            x_natural=inputs,
                            y=labels,
                            optimizer=optimizer,
                            epsilon=radius,
                            step_size=1,
                            beta=0.001)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if epoch % 1 == 0:
            print(f'Epoch [{epoch+1}/{200}], Loss: {loss_sum/len(origin_loader):.4f}')
            # save model
            # judge the loss is nan

            if loss_sum < loss_sum_best or (math.isnan(loss_sum)):
                torch.save(model.state_dict(),f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt")
                loss_sum_best = loss_sum
    model.load_state_dict(torch.load(f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt",map_location=device))
    model.eval()
    end = time.time()
    train_time = end - start
    # te
    rsr = 0
    with torch.no_grad():
        output = model(repair_data)
        rsr = (output.argmax(dim=1) == repair_label).sum().item()
        print(f"repair success rate {rsr/len(repair_data)}")
        # attack
        asr = 0
        output = model(attack_data)
        asr = (output.argmax(dim=1) == attack_label).sum().item()
        print(f"attack success rate {asr/len(attack_data)}")
        # test
        tsr = 0
        output = model(test_data)
        tsr = (output.argmax(dim=1) == test_label).sum().item()
        print(f"test success rate {tsr/len(test_data)}")
    with open(f'./tools/mnist/trade-mnist/result.txt','a') as f:
        f.write(f"For {net} {data_num} {radius} : repair:{rsr/len(repair_data)}, attack:{asr/len(attack_data)}, test:{tsr/len(test_data)}, time:{train_time}, epoch:{epoch_n} \n")




def autoattack_adv_training(net, data_num, device,radius = 0.1):
    
    print(f'Using {device} device')
    if net == 'CNN_small':
        model1 = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model1 = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model1 = FNN_small_NeuralNet().to(device)
    model1.load_state_dict(torch.load(f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt",map_location=device))

    # model1 = adv_training(net,radius, data_num=data_num, device=device, radius_bit=radius_bit)
    model1.eval()
    datas,labels = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
    datas = datas[:data_num]
    labels = labels[:data_num]
    from torchattacks import AutoAttack
    at1 = AutoAttack(model1, norm='Linf', eps=radius, version='standard', verbose=False)


    p1 = 0
    for ith, (image, label) in enumerate(zip(datas,labels)):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        adv_images1 = at1(image, label)
    
        if model1(adv_images1).argmax(dim=1)!= label:
            print("success1")
            p1 += 1
        else:
            print("fail")
    with open(f'./results/mnist/repair/autoattack/compare_autoattack_ac.txt','a') as f:
        f.write(f"For {net} {data_num} {radius} : adv-train:{p1}\n")






def patch_label_autoattack(net, patch_format, 
                            radius, repair_number, device):
    from common.repair_moudle import Netsum
    from DiffAbs.DiffAbs import deeppoly
    from mnist.mnist_utils import MnistNet_CNN_small,MnistNet_FNN_small, MnistNet_FNN_big, Mnist_patch_model,MnistProp
    device = device if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    if net == 'CNN_small':
        # model1 = CNN_small_NeuralNet().to(device)
        orinet = MnistNet_CNN_small(dom=deeppoly)
    elif net == 'FNN_big':
        # model1 = FNN_big_NeuralNet().to(device)
        orinet = MnistNet_FNN_big(dom=deeppoly)
    elif net == 'FNN_small':
        # model1 = FNN_small_NeuralNet().to(device)
        orinet = MnistNet_FNN_small(dom=deeppoly)

    orinet.to(device)
    patch_lists = []
    for i in range(10):
        if patch_format == 'small':
            patch_net = Mnist_patch_model(dom=deeppoly, name = f'small patch network {i}')
        elif patch_format == 'big':
            patch_net = Mnist_patch_model(dom=deeppoly,name = f'big patch network {i}')
        patch_net.to(device)
        patch_lists.append(patch_net)
    model2 =  Netsum(deeppoly, target_net = orinet, patch_nets= patch_lists, device=device)
    model2.load_state_dict(torch.load(f"./model/mnist_label_format/Mnist-{net}-repair_number{repair_number}-rapair_radius{radius}-{patch_format}.pt",map_location=device))

    datas,labels = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)
    # return
    
    datas = datas[:repair_number]
    labels = labels[:repair_number]


    from torchattacks import AutoAttack



    from common.prop import AndProp
    from common.bisecter import Bisecter
    repairlist = [(data[0],data[1]) for data in zip(datas, labels)]
    repair_prop_list = MnistProp.all_props(deeppoly, DataList=repairlist, input_shape= datas.shape[1:], radius= radius)
    # get the all props after join all l_0 ball feature property
    # TODO squeeze the property list, which is the same as the number of label
    all_props = AndProp(props=repair_prop_list)
    # v = Bisecter(deeppoly, all_props)
    in_lb, in_ub = all_props.lbub(device)
    in_bitmap = all_props.bitmap(device)

    bitmap = get_bitmap(in_lb, in_ub, in_bitmap, datas, device)

    p1 = 0
    p2 = 0
    p3 = 0

    for ith, (image, label) in enumerate(zip(datas,labels)):
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        # at1 = AutoAttack(model1, norm='Linf', eps=radius, version='standard', verbose=True)
        # adv_images1 = at1(image, label)
        # if model1(adv_images1).argmax(dim=1)!= label:
        #     print("success1")
        #     p1 += 1
        # else:
        #     print("fail")
        at2 = AutoAttack(model2, norm='Linf', eps=radius, version='standard', verbose=False, bitmap=bitmap)
        adv_images2 = at2(image, label)
        if model2(adv_images2, bitmap[ith]).argmax(dim=1) != label:
            print("success2")
            p2 += 1
        else:
            print("fail")
        # at3 = AutoAttack(model3, norm='Linf', eps=radius, version='standard', verbose=True)
        # adv_images3 = at3(image, label)
        # if model3(adv_images3).argmax(dim=1) != label:
        #     print("success3")
        #     p3 += 1
        # else:
        #     print("fail")
        
        # step1, ori_acc = pgd1.forward_sumsteps(image,label)
        # step2, repair_acc = pgd2.forward_sumsteps(image,label, device=device, bitmap = [in_lb, in_ub, in_bitmap])
        # step3, adt_acc = pgd3.forward_sumsteps(image,label)
        # ori_step += step1
        # repair_step += step2
        # pgd_step += step3
        # if ori_acc == 1:
        #     p1 += 1
        # if repair_acc == 1:
        #     p2 += 1
        # if adt_acc == 1:
        #     p3 += 1
            
    
    # print(f"ori_step {ori_step}, repair_step {repair_step}, pgd_step {pgd_step} \\ ori:{p1}, patch:{p2}, adv-train:{p3}")
    # with open(f'./results/mnist/repair/autoattack/compare_autoattack_ac.txt','a') as f:
    #     f.write(f"For {net} {radius} {data} {patch_format}: \\  ori:{p1}, patch:{p2}, adv-train:{p3} \\ \n")
    with open(f'./results/mnist/repair/autoattack/compare_autoattack_ac.txt','a') as f:
        f.write(f"For {net} {radius} {repair_number} : {patch_format}_label:{p2}\n")

def trades_generalization(net, radius, data_num, device):
    if net == 'CNN_small':
        model = CNN_small_NeuralNet().to(device)
    elif net == 'FNN_big':
        model = FNN_big_NeuralNet().to(device)
    elif net == 'FNN_small':
        model = FNN_small_NeuralNet().to(device)
    model.load_state_dict(torch.load(f"./tools/mnist/trade-mnist/best_{net}_{radius}_{data_num}.pt",map_location=device))

    # load data
    test_data, test_label = torch.load('./data/MNIST/processed/test_norm00.pt',map_location=device)
    # origin_data,origin_label = torch.load(f'./data/MNIST/processed/origin_data_{net}_{radius}.pt',map_location=device)


    # dataloader
    test_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    test_loader = DataLoader(test_dataset, batch_size=32)
    correct_sum = 0
    from torchattacks import AutoAttack
    attack = AutoAttack(model, norm='Linf', eps=radius, version='standard', verbose=False)
    for test_data, test_label in test_loader:
        test_data, test_label = test_data.to(device), test_label.to(device)
        adv_images = attack(test_data, test_label)
        outs = model(adv_images)
        predicted = outs.argmax(dim=1)
        correct = (predicted == test_label).sum().item()
        correct_sum += correct
        print(f"correct {correct}")
    
    with open(f'./results/mnist/repair/generalization/compare/compare_generalization.txt','a') as f:
        f.write(f"For net: {net}, radius :{radius}, repair_number: {data_num}, adv_training:{correct_sum/len(test_dataset)}\n")


if __name__ == "__main__":
    # model = train(net='FNN_big',epoch_num=20)
    # test(net='FNN_big')
    # pgd_attack(net='FNN_small')
    # stack()
    # compare_pgd_step_length(radius=0.3,repair_number=1000)

    # for data in [1000]:
    # autoattack_adv_training(net='CNN_small', data_num=200, device='cuda:0',radius = 0.05)
    for data in [50,100, 200, 500,1000]:
        for radius in [0.05, 0.1, 0.3]:
            # for radius in [0.3]:
            # if (data == 200 and radius == 0.05): # or (data == 200 and radius == 0.1):
            #     continue    

            for net in ['FNN_small','FNN_big', 'CNN_small']:
            # for net in [ 'CNN_small']:
                # for patch_format in ['small', 'big']:
                #     patch_label_autoattack(net, patch_format, radius, data,device='cuda:0')
                # adv_training_test(net, radius, data,device='cuda:0')
                # for epoch in [200]:
                    # adv_training_test_pgd(net, radius, data_num=data, device='cuda:0',epoch_n=epoch)
                    # adv_training_test(net, radius, data_num=data, device='cuda:0',epoch_n=epoch)
                trades_generalization(net, radius, data, device='cuda:2')
            
                # autoattack_adv_training(net, data,device='cuda:0',radius = radius)
                    # compare_pgd_step_length(net, patch_format, radius, data)
                    # compare_autoattack(net, patch_format, radius, data)
                    # adv_training_test(net, radius,device='cuda:0')
    #         pgd_get_data(radius=radius,multi_number=10,data_num=data,general = True)
            # pgd_get_data(net=net,radius=radius,multi_number=10,data_num=1000)
            # grad_none(net = net,radius = radius)
    # get_trainset_norm00()
            # compare_pgd_step_length(radius=radius,repair_number=data)

