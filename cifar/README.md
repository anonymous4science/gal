# Learning to Model Remainder in Supervised Learning
Code and scripts for experiments on CIFAR-10 and CIFAR-100. 

## Requirements and Environment

* Install [EfficientNet PyTorch (0.6.3+)](https://github.com/lukemelas/EfficientNet-PyTorch)
```bash
pip install efficientnet_pytorch
```
* Download [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html), and then accordingly configure variable DATA_DIR in train_cifar_base.sh and train_cifar_gal.sh.

* The code is based on [DCL](https://github.com/luoyan407/congruency) and tested with 2 Nvidia Tesla v100 (32GB).

## Usage

Run experiments with vanilla optimization methods
```bash
./train_cifar_base.sh cifar10 sgd
./train_cifar_base.sh cifar100 sgd
```

Run experiments with GAL
```bash
./train_cifar_gal.sh cifar10 sgd '100 32 16' 1 0.001
./train_cifar_gal.sh cifar100 sgd '256 64 32' 1 0.01
```

## Performance and Pretrained Models
Error rates (%) yielded by EfficientNet-B1 with various optimization methods are as follows. The correspondingly pretrained models can be found in this shared google drive [folder](https://drive.google.com/drive/folders/1ENDvQH1G8Sm3PVdFlKREKxeskRy6_tje?usp=sharing).

**CIFAR-10**:

| Optimizer  | Standard | GAL |
| ------------- | ------------- | ------------- |
| SGD  | 1.86  | 1.78 |
| Lookahead (SGD)  | 2.03  | 1.89 |
| Adabound  | 3.17  | 3.03 |

**CIFAR-100**:

| Optimizer  | Standard | GAL |
| ------------- | ------------- | ------------- |
| SGD  | 11.89  | 11.26 |
| Lookahead (SGD)  | 11.72  | 11.42 |
| Adabound  | 14.45  | 14.06 |