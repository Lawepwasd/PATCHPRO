# ADVREPAIR
 ADVREPAIR: Provable Repair of Adversarial Attack

## Requirements
### Installation
```pip install -r requirements.txt```
### Preparation
For the reproduction of the results, you need to substitute some files in the `torchattacks` package with the files in the `AutoAttack_for_AdvRepair` folder.
### Dataset and Models
you can download the dataset and models from [here]([http](https://drive.google.com/drive/folders/16XSk9CfwCnbygDfTACA_2yFmBe7vauCY?usp=drive_link)) and put it in the `data` and `models` folder respectively.

### MNIST
#### Reproduct repairing adversarial attacks on Mnist

You can run the following command:
```
python mnist/exp_mnist.py
```
#### Reproduct testing the generalization and defense against new adversarial attacks
you can run the following command:
```
python mnist/exp_mnist_generalization.py
```

### CIFAR10
#### Reproduct repairing adversarial attacks on Cifar10
You can run the following command:
```
python cifar10/exp_cifar10_feature.py
```
#### Reproduct testing the generalization and defense against new adversarial attacks
you can run the following command:
```
python cifar10/exp_cifar10_generalization.py
```
### TINYIMAGENET
#### Reproduct repairing adversarial attacks on TinyImagenet
You can run the following command:
```
python tinyimagenet/exp_tinyimagenet_feature.py
```
#### Reproduct testing the generalization and defense against new adversarial attacks
you can run the following command:
```
python tinyimagenet/exp_tinyimagenet_generalization.py
```

### Acasxu
#### Reproduct repairing property-2 on Acasxu
You can run the following command:
```
python acasxu/exp_acas.py
```
