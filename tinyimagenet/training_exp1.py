# Test WRN and ResNet on Tiny ImageNet

from tinyimagenet_prepare import training

if __name__ == '__main__':
    for model in ['resnet50', 'wide_resnet50_2']:
        training(model)
    for model in ['resnet101', 'wide_resnet101_2']:
        training(model)