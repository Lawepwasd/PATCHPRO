from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock,Bottleneck


class Wide_Resnet_model(ResNet):
    def __init__(self, BasicBlock, width_per_group) -> None:
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=200,width_per_group=width_per_group)


def Wide_Resnet_model_34_10() -> Wide_Resnet_model:
    return Wide_Resnet_model(BasicBlock=BasicBlock, width_per_group=64*10)

# def Wide_Resnet_model_50_10() -> Wide_Resnet_model:
#     return Wide_Resnet_model(Bottleneck=Bottleneck, width_per_group=10)

# def Wide_Resnet_model_34_20() -> Wide_Resnet_model:


