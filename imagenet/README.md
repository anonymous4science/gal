# Learning to Model Remainder in Supervised Learning
Code and scripts for experiments on ImageNet. 

## Requirements and Environment

* The code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and tested with 8 Nvidia Tesla v100 (16GB). 

* Follow [pytorch-image-models](https://github.com/seraphlabs-ca/SentenceMIM-demo) to install the required packages. Make sure that NVIDIA Apex is installed and AMP mixed-precision is used.

* Download [ImageNet](http://image-net.org/download). Re-organize the directory structure of the validation as follows and configure variable DATA in train.sh and train_gal.sh accordingly.
```
imagenet/images
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
│   ...
└── val
    ├── n01440764
    ├── n01443537
    ├── n01484850
    ├── n01491361
    ├── n01494475
    ...
```

## Usage

Run experiments with the standard optimization process
```bash
./train.sh
```

Run experiments with GAL
```bash
./train_gal.sh
```

## Performance and Pretrained Models
Accuracies (%) yielded by EfficientNet-B2 are as follows. The correspondingly pretrained models can be found in this shared google drive [folder](https://drive.google.com/drive/folders/1ENDvQH1G8Sm3PVdFlKREKxeskRy6_tje?usp=sharing).

| Method  | Top-1 | Top-5 |
| ------------- | ------------- | ------------- |
| Standard  | 77.89  | 93.91 |
| GAL  | 78.04  | 93.95 |