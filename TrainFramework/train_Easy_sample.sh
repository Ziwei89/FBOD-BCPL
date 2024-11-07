#!/bin/bash
#################################
p_data_root_path="/home/newdisk/ziwei/FBD-SV-2024/"
p_total_Epoch=50
p_Add_name="20241107"
p_Batch_size=8

#### Shuffle train to two datasets #######################
python3 shuffle_train_to_two_datasets.py \
        --Add_name=$p_Add_name
#### Train the modelA 50 epoches with subsetAllA dataset under Normal mode ##########################
python3 train_AP50.py \
        --data_root_path=$p_data_root_path \
        --data_augmentation \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="Easy_sample" \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=50 \
        --total_Epoch=$p_total_Epoch \
        --input_train_val_loss_ap50_str="None/None/None/0"

#### Train the modelB 50 epoches with subsetAllB dataset under Normal mode ##########################
python3 train_AP50.py \
        --data_root_path=$p_data_root_path \
        --data_augmentation \
        --data_subset="subsetAllB" \
        --modelAorB="modelB" \
        --learn_mode="Easy_sample" \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=50 \
        --total_Epoch=$p_total_Epoch \
        --input_train_val_loss_ap50_str="None/None/None/0"