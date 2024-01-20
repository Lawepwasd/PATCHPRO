import sys
from pathlib import Path
from typing import List, Optional, Tuple, Iterable, Sequence, Union
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor, nn
from DiffAbs.DiffAbs import AbsDom, AbsEle

from acas.acas_utils import AcasNet
from mnist.mnist_utils import MnistNet_FNN_small, MnistNet_FNN_big, MnistNet_CNN_small

class SupportNet(nn.Module):
    '''
    Construct the support network for repair.
    (provisional:)
    The construction of it is full connection network. Its input is the input of neural networks.
    '''
    def __init__(self, input_size: int, dom :AbsDom, hidden_sizes: List[int],
                name: str, output_size: int ) -> None:
        '''
        :param hidden_sizes: the size of all hidden layers
        :param output_size: Due to the support network is characteristic function, the output of support network should be the number of properties to be repaired.
        :param name: the name of this support network; maybe the repairing property belonging to it in the later
        '''
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = len(hidden_sizes) + 1

        # abstract domain
        self.acti = dom.ReLU()

        # concrete domain
        # self.acti = nn.ReLU()

        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            # abstract domain
            self.all_linears.append(dom.Linear(in_size, out_size))
            # concrete domain
            # self.all_linears.append(nn.Linear(in_size, out_size))

        # this layer is to judge whether the property is violated; but it will lead the discontinuity of the NN
        # self.violate_judge_layer = nn.Linear(self.hidden_sizes[-1], 2)
        # add a sigmoid layer to the end of the network
        self.sigmoid = dom.Sigmoid()
        # self.softmax = dom.Softmax()
        return 
    
    def forward(self, x):
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)
            
        
        x = self.all_linears[-1](x)
        # x = self.softmax(x)
        x = self.sigmoid(x)
        # violate_score = self.violate_judge_layer(x)

        return x
        
    
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- SupportNet ---',
            'Name: %s' % self.name,
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Activation: %s' % self.acti,
            'last function: %s' % self.sigmoid,
            '--- End of SupportNet ---'
        ]
        return '\n'.join(ss)



class PatchNet(nn.Module):
    '''
    Construct the patch network for repair.
    1. The Patchnet and Supportnet has one-to-one correspondence
    (provisional:)
    The construction of it is full connection network. Its input is the input of neural networks.
    '''
    def __init__(self, input_size: int, dom :AbsDom, hidden_sizes: List[int],
                name: str, output_size: int ) -> None:
        '''
        :param hidden_sizes: the size of all hidden layers
        :param output_size: The patch network directly add to the output , and its input is the input of neural network. So its outputsize should be equal to the orignal outputsize
        :param name: the serial number of this support network; maybe the repairing property belonging to it in the later
        '''
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1

        # layer
        self.acti = dom.ReLU()
        self.all_linears = nn.ModuleList()
        in_sizes = [self.input_size] + self.hidden_sizes
        out_sizes = self.hidden_sizes + [self.output_size]
        for in_size, out_size in zip(in_sizes, out_sizes):
            self.all_linears.append(dom.Linear(in_size, out_size))

        
        return
    
    def forward(self, x):
        for lin in self.all_linears[:-1]:
            x = lin(x)
            x = self.acti(x)
            
        x = self.all_linears[-1](x)
        return x

    
    def __str__(self):
        """ Just print everything for information. """
        ss = [
            '--- PatchNet ---',
            'Name: %s' % self.name,
            'Num layers: %d (i.e. hidden + output, excluding input layer)' % self.n_layers,
            'Input size: %d' % self.input_size,
            'Hidden sizes (len %d): ' % len(self.hidden_sizes) + str(self.hidden_sizes),
            'Output size: %d' % self.output_size,
            'Activation: %s' % self.acti,
            '--- End of PatchNet ---'
        ]
        return '\n'.join(ss)

class Netsum(nn.Module):
    '''
    This class is to add the patch net to target net:
    
    '''
    def __init__(self, dom: AbsDom, target_net, patch_nets: List[nn.Module], device = None,
                 generalization = False, repair_direction_dict = None,
                 is_label_repaired = False):
        '''
        :params 
        '''
        super().__init__()
        self.target_net = target_net
        # self.support_net = support_nets
        
        # for support, patch in zip(support_nets, patch_nets):
        #     assert(support.name == patch.name), 'support and patch net is one-to-one'

        # self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_patch_lists = len(self.patch_nets)
        self.generalization = generalization
        self.repair_direction_dict = repair_direction_dict
        self.is_label_repaired = is_label_repaired

        if device is not None:
            for i,patch in enumerate(self.patch_nets):
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        # initial bitmap
        self.bitmap = None
        
        # self.sigmoid = dom.Sigmoid()
        # self.connect_layers = []
    def set_bitmap(self, bitmap):
        self.bitmap = bitmap
    def set_generalization(self, generalization):
        self.generalization = generalization
    def set_repair_direction_dict(self, repair_direction_dict):
        self.repair_direction_dict = repair_direction_dict
    def get_bitmap(self, original_out):
        with torch.no_grad():
            # get the max and runner-up score of out
            _,index = torch.topk(original_out,2,dim = -1)
            # max_index = index[...,0]
            # runnerup_index = index[...,1]
            # match with repair_direction_dict : (,2)
            # index_clone = index.clone().unsqueeze_(1).expand(index.shape[0],self.repair_direction_dict.shape[0], index.shape[-1])
            index_clone = index.clone().unsqueeze_(1).expand(index.shape[0], len(self.patch_nets), index.shape[-1])

            is_in = torch.eq(index_clone,self.repair_direction_dict).all(dim = -1)

            is_in_test_zero = is_in.sum(dim = -1) == 0

            is_in_test_zero_index = is_in_test_zero.nonzero(as_tuple=True)[0]

            is_in_zero_assign = index[is_in_test_zero_index][...,0].unsqueeze_(1).expand(-1, self.repair_direction_dict.shape[0])

            is_in_zero_assign_map = torch.eq(is_in_zero_assign, self.repair_direction_dict[...,0])
            is_in[is_in_test_zero] = is_in_zero_assign_map
            is_in = is_in.to(torch.uint8)
            self.bitmap = is_in 
        return self.bitmap
            

    def get_bitmap_label(self, original_out):
        with torch.no_grad():
            # get the max and runner-up score of out
            # _,index = torch.topk(original_out,2,dim = -1)
            index = original_out.argmax(dim = -1)
            is_in = torch.zeros((index.shape[0], len(self.patch_nets)), dtype = torch.uint8, device = index.device)
            is_in[torch.arange(index.shape[0]), index] = 1




            self.bitmap = is_in 
        return self.bitmap
    def forward(self, x, in_bitmap = None, out = None):
        if out == None:
            out = self.target_net(x)
        else:
            assert isinstance(out, AbsEle), 'out should be AbsEle'



        if self.generalization and self.bitmap is None and self.repair_direction_dict is not None:
            if self.is_label_repaired:
                in_bitmap = self.get_bitmap_label(out)
            else:
                in_bitmap = self.get_bitmap(out)

            
        # classes_score, violate_score = self.support_net(x) # batchsize * repair_num * []
        # n_prop = in_bitmap.shape[-1]
        if in_bitmap is not None:
            for i,patch in enumerate(self.patch_nets):
                bits = in_bitmap[..., i]
                if not bits.any():
                # no one here needs to obey this property
                    continue

                ''' The default nonzero(as_tuple=True) returns a tuple, make scatter_() unhappy.
                    Here we just extract the real data from it to make it the same as old nonzero().squeeze(dim=-1).
                '''
                bits = bits.nonzero(as_tuple=True)[0]
                if isinstance(out, Tensor):
                    p_out = patch(x)
                    out[bits] += p_out[bits]
                elif isinstance(out, AbsEle):
                    replace_item = out[bits] + patch(x[bits]) # may not only one prop
                    out.replace(in_bitmap[..., i], replace_item)
        return out
        
        

    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- InputNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)



# class NetOnlySumPatch(Netsum):
#     def forward(self, x, in_bitmap, out):
#         for i,patch in enumerate(self.patch_nets):
#             bits = in_bitmap[..., i]
#             if not bits.any():
#             # no one here needs to obey this property
#                 continue

#             ''' The default nonzero(as_tuple=True) returns a tuple, make scatter_() unhappy.
#                 Here we just extract the real data from it to make it the same as old nonzero().squeeze(dim=-1).
#             '''
#             bits = bits.nonzero(as_tuple=True)[0]
#             if isinstance(x, Tensor):
#                 out[bits] += patch(x[bits])
#             elif isinstance(x, AbsEle):
#                 replace_item = out[bits] + patch(x[bits]) # may not only one prop
#                 out.replace(in_bitmap[..., i], replace_item)
#         return out
#         pass

# class NetFeatureSum(nn.Module):
#     '''
#     This class is to add the patch net to target net:
    
#     '''
#     def __init__(self, dom: AbsDom, target_net, patch_nets: List[nn.Module], device = None,):
#         '''
#         :params 
#         '''
#         super().__init__()
#         self.target_net = target_net

#         self.patch_nets = patch_nets
#         self.acti = dom.ReLU()
#         self.len_patch_lists = len(self.patch_nets)

#         if device is not None:
#             for i,patch in enumerate(self.patch_nets):
#                 self.add_module(f'patch{i}',patch)
#                 patch.to(device)



#     def forward(self, x):
#         # split the target net to two parts, 1st part is the feature extractor, 2nd part is the classifier without sigmoid
#         self.model1,self.model2 = self.target_net.split()
#         feature = self.model1(x)
#         out = self.model2(feature)

#         classes_score = self.support_net(feature) # batchsize * repair_num_propertys
        


#         for i,patch in enumerate(self.patch_nets):
#             # violate_score[...,0] is the score of safe, violate_score[...,1] is the score of violate
#             # we repair the property according to the violate score
#             pa = patch(feature)
#             if isinstance(pa, Tensor):
#                 K = pa.norm(p = float('inf'),dim = -1).view(-1,1)
#                 K = K.detach()
#                 bar = (K * classes_score[:,i].view(-1,1))
#                 out += self.acti(pa + bar - self.A*K)\
#                     + -1*self.acti(-1*pa + bar -self.A*K )
                


#             else:
#                 K = pa.ub().norm(p = float('inf'),dim = -1).view(-1,1)

#                 # avoid multiply grad
#                 # K.requires_grad_(False)
#                 K = K.detach()

#                 bar = (K * classes_score[:,:,i])
#                 bar = bar.unsqueeze(dim = 2).expand_as(pa)
#                 # using the upper bound of the patch net to instead of the inf norm of patch net
#                 out += self.acti(pa + bar + (-self.A*K.unsqueeze(-1).expand_as(pa._lcnst)) )\
#                     + -1*self.acti(-1*pa + bar + (-self.A*K.unsqueeze(-1).expand_as(pa._lcnst)))
                
#         # origin add patch repair, then sigmoid
#         # out = self.sigmoid(origin_before_sigmoid)
#         return out
    
#     def __str__(self):
#         """ Just print everything for information. """
#         # TODO information for each support and patch net as components
#         ss = [
#             '--- FeatureNetSum ---',
#             'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
#             'Input size: %d' % self.target_net.input_size,
#             'Output size: %d' % self.target_net.output_size,
#             'Threshold value: %d' % self.k,
#             '--- End of IntersectionNetSum ---'
#         ]
#         return '\n'.join(ss)

class NetFeatureSumPatch(nn.Module):
    def __init__(self, feature_extractor ,feature_sumnet: Netsum):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_sumnet = feature_sumnet

    def forward(self, x, bitmap = None):
        feature = self.feature_extractor(x)
        out = self.feature_sumnet(feature, bitmap)
        return out

class IntersectionNetSum(nn.Module):
    '''
    This class is the complement of single-region repair in REASSURE.The function is:
    
     h_{\mathcal{A}}(x, \gamma)=\sigma\left(p_{\mathcal{A}}(x)+K \cdot g_{\mathcal{A}}(x, \gamma)-K\right)-\sigma\left(-p_{\mathcal{A}}(x)+K \cdot g_{\mathcal{A}}(x, \gamma)-K\right)
    '''
    def __init__(self, dom: AbsDom, target_net: AcasNet, support_nets : List[nn.Module], patch_nets: List[nn.Module], device = None, ):
        '''
        :params K : we define the threshold value k as 1e8 as default.
        '''
        super().__init__()
        self.target_net = target_net
        
        for support, patch in zip(support_nets, patch_nets):
            assert(support.name == patch.name), 'support and patch net is one-to-one'

        self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_support_lists = len(self.support_nets)
        self.K = [0.01 for i in range(self.len_support_lists)] # initial

        if device is not None:
            for i,support, patch in zip(range(len(self.support_nets)),self.support_nets, self.patch_nets):
                self.add_module(f'support{i}',support)
                support.to(device)
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        
        
        # self.connect_layers = []



    def forward(self, x):
        out = self.target_net(x)
        for i, support, patch in zip(range(self.len_support_lists),self.support_nets, self.patch_nets):
            out += self.acti(patch(x) + self.K[i]*support(x) - self.K[i]) \
                - self.acti(-1*patch(x) + self.K[i]*support(x) - self.K[i])
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- IntersectionNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)


class ConnectionNetSum(nn.Module):
    '''
    This class is to directly connect the support net and patch net:
    
    '''
    def __init__(self, dom: AbsDom, target_net: AcasNet, support_nets : List[nn.Module], patch_nets: List[nn.Module], device = None, ):
        '''
        :params
        '''
        super().__init__()
        self.target_net = target_net
        
        for support, patch in zip(support_nets, patch_nets):
            assert(support.name == patch.name), 'support and patch net is one-to-one'

        self.support_nets = support_nets
        self.patch_nets = patch_nets
        self.acti = dom.ReLU()
        self.len_support_lists = len(self.support_nets)

        if device is not None:
            for i,support, patch in zip(range(len(self.support_nets)),self.support_nets, self.patch_nets):
                self.add_module(f'support{i}',support)
                support.to(device)
                self.add_module(f'patch{i}',patch)
                patch.to(device)
        
        
        # self.connect_layers = []



    def forward(self, x):
        out = self.target_net(x)
        for i, support, patch in zip(range(self.len_support_lists),self.support_nets, self.patch_nets):
            out += self.acti(patch(x) + self.K[i]*support(x) - self.K[i]) \
                - self.acti(-1*patch(x) + self.K[i]*support(x) - self.K[i])
        return out
    
    def __str__(self):
        """ Just print everything for information. """
        # TODO information for each support and patch net as components
        ss = [
            '--- IntersectionNetSum ---',
            'Num net: support %d , patch %d' % (len(self.support_nets),len(self.patch_nets)),
            'Input size: %d' % self.target_net.input_size,
            'Output size: %d' % self.target_net.output_size,
            'Threshold value: %d' % self.k,
            '--- End of IntersectionNetSum ---'
        ]
        return '\n'.join(ss)
    


# class Netsum(nn.Module):
#     '''
#     This class is to add the patch net to target net:
    
#     '''
#     def __init__(self, dom: AbsDom, target_net: AcasNet, support_nets: nn.Module, patch_nets: List[nn.Module], A, device = None ):
#         '''
#         :params 
#         '''
#         super().__init__()
#         self.target_net = target_net
#         self.support_net = support_nets
        
#         # for support, patch in zip(support_nets, patch_nets):
#         #     assert(support.name == patch.name), 'support and patch net is one-to-one'

#         # self.support_nets = support_nets
#         self.patch_nets = patch_nets
#         self.acti = dom.ReLU()
#         self.len_patch_lists = len(self.patch_nets)

#         if device is not None:
#             for i,patch in enumerate(self.patch_nets):
#                 self.add_module(f'patch{i}',patch)
#                 patch.to(device)
#         self.A = A
        
#         # self.sigmoid = dom.Sigmoid()
#         # self.connect_layers = []



#     def forward(self, x):
#         out = self.target_net(x)
#         # classes_score, violate_score = self.support_net(x) # batchsize * repair_num * []
#         # classes_score = self.support_net(x) # batchsize * repair_num_propertys

#         # we should make sure that the violate_score is not trainable, otherwise the net will not linear
#         # violate_score.requires_grad_(False)

#         # compute the K in reassure
#         # norms = torch.norm(classes_score, p=float('inf'), dim=1)
#         # norms.requires_grad_(False)

#         # for i,patch in enumerate(self.patch_nets):
#         #     # violate_score[...,0] is the score of safe, violate_score[...,1] is the score of violate
#         #     # we repair the property according to the violate score
#         #     pa = patch(x)
#         #     if isinstance(pa, Tensor):
#         #         K = pa.norm(p = float('inf'),dim = -1).view(-1,1)
#         #         K = K.detach()
#         #         bar = (K * classes_score[:,i].view(-1,1))
#         #         # -K, -K
#         #         out += self.acti(pa + bar - self.A*K)\
#         #             + -1*self.acti(-1*pa + bar -self.A*K )
#         #     else:
#         #         K = pa.ub().norm(p = float('inf'),dim = -1).view(-1,1)

#         #         # avoid multiply grad
#         #         # K.requires_grad_(False)
#         #         K = K.detach()

#         #         bar = (K * classes_score[:,:,i])
#         #         bar = bar.unsqueeze(dim = 2).expand_as(pa)
#         #         # using the upper bound of the patch net to instead of the inf norm of patch net
#         #         # + (-1*K.unsqueeze(-1).expand_as(pa._lcnst)), + (-1*K.unsqueeze(-1).expand_as(pa._lcnst))
#         #         out += self.acti(pa + bar + (-self.A*K.unsqueeze(-1).expand_as(pa._lcnst)) )\
#         #             + -1*self.acti(-1*pa + bar + (-self.A*K.unsqueeze(-1).expand_as(pa._lcnst)))
                
#         # out = self.sigmoid(out)
#         # return out

#     # Directly multiply
#     def forward(self, x):
#         out = self.target_net(x)
#         # classes_score, violate_score = self.support_net(x) # batchsize * repair_num * []
#         classes_score = self.support_net(x) # batchsize * repair_num_propertys

#         for i,patch in enumerate(self.patch_nets):
#             # violate_score[...,0] is the score of safe, violate_score[...,1] is the score of violate
#             # we repair the property according to the violate score
#             pa = patch(x)
#             if isinstance(pa, Tensor):
#                 classes_score_entry = classes_score[:,i].unsqueeze(dim = 1).expand_as(pa)
#                 out += pa * classes_score_entry
#             else:
#                 classes_score_entry = classes_score[:,:,i].unsqueeze(dim = 2).expand_as(pa)
#                 out += pa * classes_score_entry
                
#         # out = self.sigmoid(out)
#         return out

def get_bitmap(in_lb: Tensor, in_ub: Tensor, in_bitmap: Tensor, batch_inputs: Tensor,device = 'cuda'):
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
        bitmap = torch.zeros((batch_inputs.shape[0], in_bitmap.shape[1]), device = device).to(torch.uint8)
        bitmap_i, inbitmap_j =  is_in.nonzero(as_tuple=True)
        if bitmap_i.shape[0] != 0:
            bitmap[bitmap_i, :] = in_bitmap[inbitmap_j, :]
        else:
            pass


        return bitmap