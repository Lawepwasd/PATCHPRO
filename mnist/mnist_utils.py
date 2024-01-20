'''
MNist task setup, includes :1. the class Photo property and its child(Mnist) 2. the class MNist net
'''
import datetime
import enum
import sys
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
from abc import ABC, abstractclassmethod

import torch
from torch import Tensor, nn
from DiffAbs.DiffAbs import AbsDom, AbsEle
import numpy as np
import ast
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.prop import OneProp, AndProp
from common.utils import sample_points

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class PhotoProp(OneProp):
    '''
    Define a mnist property from input
    '''
    def __init__(self, input_shape: Tensor, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        super().__init__(name, dom, safe_fn, viol_fn, fn_args)
        '''
        :param input_dimension: the dimension of input, (channel, height, width)
        '''
        self.input_dimension = input_shape
        # the input bounds of the input is 3 dimension, (channel, height, width)
        # initialize the input bounds as empty
        self.lower_bounds = torch.zeros(*input_shape).to(device)
        self.upper_bounds = torch.ones(*input_shape).to(device)
        self.reset_input_bound()
        
        # for i, j,k in product(range(input_dimension[0]), range(input_dimension[1]), range(input_dimension[2])):
        #     self.input_bounds[i][j][k] = (-1000000, 1000000)
    def reset_input_bound(self):
        self.input_bounds = (self.lower_bounds, self.upper_bounds)
        
    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        # bs = torch.tensor(self.input_bounds)
        # bs = bs.unsqueeze(dim=0)
        # lb, ub = bs[..., 0], bs[..., 1]
        lb, ub = self.input_bounds

        if device is not None:
            lb, ub = lb.unsqueeze(dim=0).to(device), ub.unsqueeze(dim=0).to(device)
        return lb, ub

    def set_input_bound(self, new_low: Tensor = None, new_high: Tensor = None):
        if new_low is not None:
            new_low = new_low.to(device)
            self.lower_bounds = new_low
        if new_high is not None:
            new_high = new_high.to(device)
            self.upper_bounds = new_high
        assert(torch.all(self.lower_bounds <= self.upper_bounds))

        self.reset_input_bound()
        return


class FeatureProp(OneProp):
    '''
    Define a feature property
    '''
    def __init__(self, input_dimension: int, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        super().__init__(name, dom, safe_fn, viol_fn, fn_args)
        self.input_dimension = input_dimension
        self.input_bounds = [(-1000000, 1000000) for _ in range(input_dimension)]
    
    def lbub(self, device=None) -> Tuple[Tensor, Tensor]:
        """ Return <LB, UB>, both of size <1xDim0>. """
        bs = torch.tensor(self.input_bounds)
        bs = bs.unsqueeze(dim=0)
        lb, ub = bs[..., 0], bs[..., 1]
        if device is not None:
            lb, ub = lb.to(device), ub.to(device)
        return lb, ub
    
    def set_input_bound(self, idx: int, new_low: float = None, new_high: float = None):
        low, high = self.input_bounds[idx]
        if new_low is not None:
            low = max(low, new_low)

        if new_high is not None:
            high = min(high, new_high)

        assert(low <= high)
        self.input_bounds[idx] = (low, high)
        return


class MnistProp(PhotoProp):
    LABEL_NUMBER = 10

    class MnistOut(enum.IntEnum):
        '''
        the number from 0-9
        '''
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9
    
    def __init__(self, input_shape: Tensor, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        '''
        :param input_shape: the dimension of input, (channel, height, width)
        '''
        super().__init__(input_shape, name, dom, safe_fn, viol_fn, fn_args)
    
    @classmethod
    def attack_input(cls, dom: AbsDom, input_shape: int, data: Tensor, label: int, radius: float,
                     number: int = 1):
        '''
        The mnist property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = MnistProp(name=f'attack_input{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label])  # mean/range hardcoded 
        
        p.set_input_bound(new_low=data - radius)
        p.set_input_bound(new_high=data + radius)     

        return p
    
    @classmethod
    def attack_input_label(cls, dom: AbsDom, input_shape: int, data: Tensor, label: int, radius: float,
                     number: int = 1):
        '''
        The mnist property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = MnistProp(name=f'attack_input_label{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label])  # mean/range hardcoded 
        
        p.set_input_bound(new_low=data - radius)
        p.set_input_bound(new_high=data + radius)     

        return p



    @classmethod
    def feature_input(cls, dom: AbsDom, input_shape: int, lb: Tensor, ub: Tensor, label: int,
                     number: int = 1):
        '''
        :params input_shape: the input/feature dimension
        :params label: the output which should be retained
        :params lb: the lower bound of the feature
        :params ub: the upper bound of the feature
        '''
        p = MnistProp(name=f'feature_input{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label]) 
        p.set_input_bound(new_low=lb, new_high=ub)
        # p.set_input_bound(new_high=ub)
        return p
    
    @classmethod
    def all_props(cls, dom: AbsDom, DataList: List[Tuple[Tensor, Tensor]], 
                  input_shape: int, radius: float = 0.1, 
                  tasktype:str = 'attack_input'):
        '''
        :param tasktype: the type of task, e.g. 'attack_input' or 'attack_input_label'in mnist repair
        :param dom: the domain of input, e.g. Deeppoly
        :param DataList: the list of data, e.g. [(data1, label1), (data2, label2), ...]
        :param input_shape: the dimension of input/feature
        :param radius: the radius of the attack input/feature
        '''

        datalen = len(DataList)
        names = [(DataList[i][0], DataList[i][1]) for i in range(datalen)]
        a_list = []
        for data,label in names:
            a = getattr(cls, tasktype)(dom, input_shape, data, label.item(), radius)
            a_list.append(a)
        
        return a_list
    
    @classmethod
    def all_feature_props(cls, dom: AbsDom, bounds: List[Tuple[Tensor, Tensor]], 
                  labels: Tensor,
                  feature_shape: int, 
                  tasktype:str = 'feature_input'):
        '''
        :param tasktype: the type of task, e.g. 'attack_input' in mnist repair
        :param dom: the domain of input, e.g. Deeppoly
        :param bounds: the list of lb and ub, e.g. [(lb1, ub1), (lb2, ub2), ...]
        :param input_shape: the dimension of input/feature
        '''
        a_list = []
        for (lb,ub),label in zip(bounds,labels):
            a = getattr(cls, tasktype)(dom, feature_shape, lb, ub, label.item())
            a_list.append(a)
        return a_list

        
        

class MnistFeatureProp(FeatureProp):
    
    LABEL_NUMBER = 10

    class MnistOut(enum.IntEnum):
        '''
        the number from 0-9
        '''
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9

    def __init__(self, input_dimension: int, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        '''
        :param input_dimension: the dimension of input/feature
        '''
        super().__init__(input_dimension, name, dom, safe_fn, viol_fn, fn_args)
    
    @classmethod
    def all_props(cls, dom: AbsDom, DataList: List[Tuple[Tensor, Tensor]], input_dimension: int, radius: float = 0.1, tasktype:str = 'attack_feature'):
        '''
        :param tasktype: the type of task, e.g. 'attack_feature' in mnist repair
        :param dom: the domain of input, e.g. Deeppoly
        :param DataList: the list of data, e.g. [(data1, label1), (data2, label2), ...]
        :param input_dimension: the dimension of input/feature
        :param radius: the radius of the attack input/feature
        '''

        datalen = len(DataList)
        names = [(DataList[i][0], DataList[i][1]) for i in range(datalen)]
        a_list = []
        for data,label in names:
            a = getattr(cls, tasktype)(dom, input_dimension, data, label.item(), radius)
            a_list.append(a)
        
        return a_list
    
    @classmethod
    def attack_feature(cls, dom: AbsDom, input_dimension: int, data: Tensor, label: int, radius: float):
        '''
        The mnist feature property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = MnistFeatureProp(name='attack_feature', input_dimension=input_dimension, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label])  # mean/range hardcoded 
        for j in range(input_dimension):
            p.set_input_bound(j, new_low=data[j].item() - radius)
            p.set_input_bound(j, new_high=data[j].item() + radius)     
                

        return p

class Mnist_patch_model(nn.Module):
    '''
    The model is to repair the mnist data from input rather than feature
    '''
    def __init__(self,dom: AbsDom, name: str):
        super(Mnist_patch_model,self).__init__()
        self.dom = dom
        self.name = name
        # self.extractor = nn.Sequential(
        #     dom.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
        #     dom.ReLU(),
        #     dom.MaxPool2d(kernel_size=2)
        # )
        self.flatten = dom.Flatten()
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        # multiply the input_shape
        if 'small' in name:
            self.extractor = nn.Sequential(
                dom.Linear(in_features=1*28*28,out_features=64),
                dom.ReLU(),
                dom.Linear(in_features=64,out_features=16),
                dom.ReLU(),
            )
            self.classifier = nn.Sequential(
                # dom.Linear(in_features=16*14*14,out_features=10)
                dom.Linear(in_features=16,out_features=10)
            )
        
        else:
            self.extractor = nn.Sequential(
                dom.Linear(in_features=1*28*28,out_features=256),
                dom.ReLU(),
                dom.Linear(in_features=256,out_features=64),
                dom.ReLU(),
            )
            self.classifier = nn.Sequential(
                # dom.Linear(in_features=16*14*14,out_features=10)
                dom.Linear(in_features=64,out_features=10)
            )
        
    def forward(self,x):
        # x = self.extractor(x)
        # x = x.view(16*14*14,-1)
        # out = self.classifier(x)
        x = self.flatten(x)
        x = self.extractor(x)
        out = self.classifier(x)
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- PatchNet ---',
            'Name: %s' % self.name,
            '--- End of PatchNet ---'
        ]
        return '\n'.join(ss)

class Mnist_feature_patch_model(nn.Module):
    def __init__(self,dom: AbsDom, name: str, input_dimension: int) -> None:
        super().__init__()
        self.dom = dom
        self.name = name
        self.input_dimension = input_dimension
        self.classifier = nn.Sequential(
            dom.Linear(in_features=input_dimension,out_features=16),
            dom.ReLU(),
            dom.Linear(in_features=16,out_features=10),
            dom.ReLU(),
        )
    def forward(self,x):
        out = self.classifier(x)
        return out
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- feature PatchNet ---',
            'Name: %s' % self.name,
            '--- End of feature PatchNet ---'
        ]
        return '\n'.join(ss)

class MnistNet_FNN_small(nn.Module):
    def __init__(self, dom: AbsDom) -> None:
        super().__init__()
        self.dom = dom
        self.flatten = dom.Flatten()
        self.fc1 = dom.Linear(784, 50)
        self.relu = dom.ReLU()

        self.fc2 = dom.Linear(50, 50)
        self.fc3 = dom.Linear(50, 50)
        self.fc4 = dom.Linear(50, 50)
        self.fc5 = dom.Linear(50, 50)

        self.fc6 = dom.Linear(50, 32)
        self.fc7 = dom.Linear(32, 10)
    
    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        
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
    
    def split(self):
        return nn.Sequential(
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.relu,
            self.fc4,
            self.relu,
            self.fc5,
            self.relu,
            self.fc6,
            self.relu
        ), nn.Sequential(
            self.fc7
        )



class MnistNet_FNN_big(nn.Module):
    def __init__(self, dom: AbsDom) -> None:
        super().__init__()
        self.dom = dom
        self.flatten = dom.Flatten()
        self.fc1 = dom.Linear(784, 200)
        self.relu = dom.ReLU()

        self.fc2 = dom.Linear(200, 200)
        self.fc3 = dom.Linear(200, 200)
        self.fc4 = dom.Linear(200, 200)
        self.fc5 = dom.Linear(200, 200)

        self.fc6 = dom.Linear(200, 32)
        self.fc7 = dom.Linear(32, 10)
    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
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

    def split(self):
        return nn.Sequential(
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3,
            self.relu,
            self.fc4,
            self.relu,
            self.fc5,
            self.relu,
            self.fc6,
            self.relu
        ), nn.Sequential(
            self.fc7
        )



class MnistNet_CNN_small(nn.Module):
    '''
    abstract module of bank, credit and census
    # :param json file: The configuration file of Fairness task in Socrates
    :param means: The means of Dataset
    :param range: The range of Dataset
    # :param inputsize: The input size of NN, which is related to Dataset

    '''
    def __init__(self, dom: AbsDom) -> None:
        super().__init__()
        self.dom = dom
        self.conv1 = dom.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        # self.conv2 = dom.Conv2d(32, 64, kernel_size=5)
        # self.maxpool = dom.MaxPool2d(2)
        self.flatten = dom.Flatten()
        self.relu = dom.ReLU()
        self.fc1 = dom.Linear(16*14*14, 100)
        self.fc2 = dom.Linear(100, 10)
        
        # self.sigmoid = dom.Sigmoid()

    def forward(self, x: Union[Tensor, AbsEle]) -> Union[Tensor, AbsEle]:
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.maxpool(x)
        # x = self.relu(x)
        # x = torch.flatten(x, 1)
        # x = x.view(x.size[0], 1024)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def split(self):
        return nn.Sequential(
            self.conv1,
            self.relu,
            # torch.flatten(x, 1),
            self.flatten,
            self.fc1,
            self.relu
        ), nn.Sequential(
            
            self.fc2,
            # self.sigmoid()
        )

    
    
    





