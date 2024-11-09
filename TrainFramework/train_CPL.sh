#!/bin/bash
############### train CPL algorithm.##################
p_data_root_path="/home/newdisk/ziwei/FBD-SV-2024/"
p_total_Epoch=100
p_Add_name="20241102"
p_Batch_size=8
p_cpl_mode="hard"
modelB_total_train_val_loss_ap50_str=None/None/None/0
modelA_total_train_val_loss_ap50_str=None/None/None/0
log_txt="./logs_txt/log_"$p_Add_name".txt"

mkdir -p logs_txt
#### Shuffle train to two datasets #######################
python3 shuffle_train_to_two_datasets.py \
        --Add_name=$p_Add_name

python3 copy_normal_model_to_custom_model.py \
        --learn_mode="CPL" \
        --cpl_mode=$p_cpl_mode \
        --Add_name=$p_Add_name

#### epoch 0 to epoch 100, every epoch update the score one time. #####
for n in {0..99};
do 
    s_epoch=$(($n))
    e_epoch=$(($n + 1))


    ## update the object weight of subsetAllA with modelB ###
    python3 update_object_weight.py \
        --data_root_path=$p_data_root_path \
        --data_subset="subsetAllA" \
        --modelAorB="modelB" \
        --learn_mode="CPL" \
        --cpl_mode=$p_cpl_mode \
        --current_epoch=$s_epoch \
        --total_Epoch=$p_total_Epoch \
        --Add_name=$p_Add_name
    
    ## update the object weight of subsetAllB with modelA ###
    python3 update_object_weight.py \
        --data_root_path=$p_data_root_path \
        --data_subset="subsetAllB" \
        --modelAorB="modelA" \
        --learn_mode="CPL" \
        --cpl_mode=$p_cpl_mode \
        --current_epoch=$s_epoch \
        --total_Epoch=$p_total_Epoch \
        --Add_name=$p_Add_name
    
    #### Train the modelB 1 epoch with subsetAllB dataset under CPL mode ##########################
    modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_root_path=$p_data_root_path \
        --data_augmentation \
        --load_pretrain_model \
        --data_subset="subsetAllB" \
        --modelAorB="modelB" \
        --learn_mode="CPL" \
        --cpl_mode=$p_cpl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --input_train_val_loss_ap50_str="$modelB_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelB_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelB_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt


    #### Train the modelA 1 epoch with subsetAllA dataset under CPL mode ##########################
    modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_root_path=$p_data_root_path \
        --data_augmentation \
        --load_pretrain_model \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="CPL" \
        --cpl_mode=$p_cpl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --input_train_val_loss_ap50_str="$modelA_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelA_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelA_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt
    
done
