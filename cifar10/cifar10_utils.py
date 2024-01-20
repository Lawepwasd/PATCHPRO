import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mnist.mnist_utils import PhotoProp
import enum

from itertools import product

from typing import List, Optional, Tuple, Iterable, Sequence, Union
import torch
from torch import Tensor, nn
from DiffAbs.DiffAbs import AbsDom, AbsEle

from common.prop import OneProp, AndProp
from common.utils import sample_points

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from vgg import VGG
# from resnet import ResNet,BasicBlock

class CifarProp(PhotoProp):
    LABEL_NUMBER = 10

    class CifarOut(enum.IntEnum):
        airplane = 0
        automobile = 1
        bird = 2
        cat = 3
        deer = 4
        dog = 5
        frog = 6
        horse = 7
        ship = 8
        truck = 9
    
    def __init__(self, input_shape: Tensor, name: str, dom: Optional[AbsDom], safe_fn: str, viol_fn: str, fn_args: Iterable):
        '''
        :param input_shape: the dimension of input, (channel, height, width)
        '''
        super().__init__(input_shape, name, dom, safe_fn, viol_fn, fn_args)
        self.input_dimension = input_shape
        # the input bounds of the input is 3 dimension, (channel, height, width)
        # initialize the input bounds as empty
        self.lower_bounds = -100*torch.ones(*input_shape).to(device)
        self.upper_bounds = 100*torch.ones(*input_shape).to(device)
        self.reset_input_bound()

    @classmethod
    def attack_input(cls, dom: AbsDom, input_shape: int, data: Tensor, label: int, radius: float,
                     number: int = 1):
        '''
        The cifar property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = CifarProp(name=f'attack_input{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label])  # mean/range hardcoded 
        
        p.set_input_bound(new_low=data - radius)
        p.set_input_bound(new_high=data + radius)     

        return p

    @classmethod
    def attack_input_label(cls, dom: AbsDom, input_shape: int, data: Tensor, label: int, radius: float,
                     number: int = 1):
        '''
        The cifar property is Data-based property. One data point correspond to one l_0 ball.
        :params input_dimension: the input/feature dimension
        :params label: the output which should be retained
        :params radius: the radius of the attack input/feature
        '''
        p = CifarProp(name=f'attack_input_label{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
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
        p = CifarProp(name=f'feature_input{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label]) 
        p.set_input_bound(new_low=lb, new_high=ub)
        # p.set_input_bound(new_high=ub)
        return p
    
    @classmethod
    def feature_input_label(cls, dom: AbsDom, input_shape: int, lb: Tensor, ub: Tensor, label: int,
                    number: int = 1):
        '''
        :params input_shape: the input/feature dimension
        :params label: the output which should be retained
        :params lb: the lower bound of the feature
        :params ub: the upper bound of the feature
        '''
        p = CifarProp(name=f'feature_input{number}', input_shape=input_shape, dom=dom, safe_fn='cols_is_max', viol_fn='col_not_max',
                    fn_args=[label]) 
        p.set_input_bound(new_low=lb, new_high=ub)
        # p.set_input_bound(new_high=ub)
        return p
    
    @classmethod
    def all_props(cls, dom: AbsDom, DataList: List[Tuple[Tensor, Tensor]], 
                  input_shape: int, radius: float = 0.1, 
                  tasktype:str = 'attack_input'):
        '''
        :param tasktype: the type of task, e.g. 'attack_input' or 'attack_input_label' in cifar10 repair
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
        :param tasktype: the type of task, e.g. 'feature_input' or 'feature_input_label'in mnist repair
        :param dom: the domain of input, e.g. Deeppoly
        :param bounds: the list of lb and ub, e.g. [(lb1, ub1), (lb2, ub2), ...]
        :param input_shape: the dimension of input/feature
        '''
        a_list = []
        for (lb,ub),label in zip(bounds,labels):
            a = getattr(cls, tasktype)(dom, feature_shape, lb, ub, label.item())
            a_list.append(a)
        return a_list




class Vgg_model(VGG):

    def __init__(self, dom: AbsDom) -> None:
        super().__init__('VGG19')
        self.dom = AbsDom
        self.classifier = dom.Linear(512, 10)
        # self.sp = dom.Linear(32, 10)
from torchvision.models.resnet import ResNet, BasicBlock

class Resnet_model(ResNet):
    def __init__(self, dom: AbsDom) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.dom = dom
        self.fc = dom.Linear(512, 10)
        # self.sp = dom.Linear(32, 10)
    def split(self):
        return nn.Sequential(
            self.conv1, 
            self.bn1, 
            self.relu,
            self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # self.linear
            ), nn.Sequential(
                self.fc
                )
    



class Cifar_feature_patch_model(nn.Module):
    def __init__(self,dom: AbsDom, name: str, input_dimension: int) -> None:
        super().__init__()
        self.dom = dom
        self.name = name
        self.input_dimension = input_dimension
        self.classifier = nn.Sequential(
            dom.Linear(in_features=input_dimension, out_features=10),

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


