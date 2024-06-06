#!/usr/bin/env bash
DEVICE=0

# echo ""
# echo "-------------------------------------------------"
# echo "| Train detail                          |"
# echo "-------------------------------------------------"
# python train_cutmix.py  --name ft_ResNet50_detail_cutmix --data_dir ../detail --height 1950



# echo ""
# echo "-------------------------------------------------"
# echo "| Train represent                          |"
# echo "-------------------------------------------------"

# python train_cutmix.py --name ft_ResNet50_represent_cutmix --data_dir ../represent --height 780


#echo " Test represent                          "

#python test.py --name ft_ResNet50_represent_cutmix --which_epoch 100
echo " Test represent 30                    "

python test.py --name ft_ResNet50_represent_cutmix --test_dir ../represent --name ft_ResNet50_represent_cutmix --which_epoch 030

echo " Test represent 50                    "

python test.py --name ft_ResNet50_represent_cutmix --test_dir ../represent --name ft_ResNet50_represent_cutmix --which_epoch 050

echo " Test detailed 30                    "

python test.py --name ft_ResNet50_detail_cutmix --test_dir ../detail --name ft_ResNet50_detail_cutmix --which_epoch 030

echo " Test detailed 500                    "

python test.py --name ft_ResNet50_detail_cutmix --test_dir ../detail --name ft_ResNet50_detail_cutmix --which_epoch 050
