# VGG11 to PlainNet5 (5x5 kernels)

## Transfer VGG11 to PlainNet5 (5x5 kernels) with the proposed heterogeneous model transfer (HTM) method
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2plain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

## Transfer VGG11 to PlainNet5 (5x5 kernels) without the proposed HTM
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2plain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint --reduce_to_baseline
```

# VGG to ResNet-8
## Transfer VGG11 to PlainNet5 (5x5 kernels) with the proposed HTM
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2resnet.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

## Transfer VGG11 to PlainNet5 (5x5 kernels) without the proposed HTM
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_vgg2resnet.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint --reduce_to_baseline
```

# Ablation studies
## shuffle HTM
Step one: In a_hetero_model_transfer.py, we uncomment line 228 and comment 229 (the transfer_from_hetero_model_more_interval function) as follows: 
uncomment "indexes = [3,1,0,7,4,6,2]" for shuffle HTM
comment "indexes = [0,2,4,6]" to turn off interval HTM

Step two: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_random_chain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

## interval HTM
Step one: In a_hetero_model_transfer.py, we uncomment line 229 and comment 228 (the transfer_from_hetero_model_more_interval function) as follows: 
comment "indexes = [3,1,0,7,4,6,2]" to turn off shuffle HTM
uncomment "indexes = [0,2,4,6]" for interval HTM

Step two: 
```bash
CUDA_VISIBLE_DEVICES=0 python train_sub_cifar100_transfer_from_ImageNet_random_chain.py --use_pretrain --save_path checkpoint5x5 --load_path checkpoint
```

# Note
More details about the person re-identification task are omitted for review simplification. In final version, we will add these code.




