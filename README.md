# HMT: Heterogeneous Model Transfer between Different Neural Networks
ICLR 2021, under review

## 1. VGG11 to PlainNet5 (5x5 kernels)

#### 1.1 Transfer VGG11 to PlainNet5 (5x5 kernels) with HMT
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2plain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

#### 1.2 Transfer VGG11 to PlainNet5 (5x5 kernels) without HMT
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2plain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint --reduce_to_baseline
```

## 2. VGG to ResNet-8
#### 2.1 Transfer VGG11 to ResNet-8 with HMT
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2resnet.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

#### 2.2 Transfer VGG11 to ResNet-8 without HMT
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2resnet.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint --reduce_to_baseline
```

## 3. Ablation studies
#### 3.1 shuffle HTM
Step one: In a_hetero_model_transfer.py, we uncomment line 228 and comment 229 (the transfer_from_hetero_model_more_interval function) as follows: 

uncomment "indexes = [3,1,0,7,4,6,2]" for shuffle HMT, 
comment "indexes = [0,2,4,6]" to turn off interval HMT

Step two: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_random_chain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

#### 3.2 interval HTM
Step one: In a_hetero_model_transfer.py, we uncomment line 229 and comment 228 (the transfer_from_hetero_model_more_interval function) as follows:

comment "indexes = [3,1,0,7,4,6,2]" to turn off shuffle HMT, 
uncomment "indexes = [0,2,4,6]" for interval HMT

Step two: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_random_chain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

## 4. Note
More details about the person re-identification task are omitted for simplifying the review. In final version, we will add these code into it.




