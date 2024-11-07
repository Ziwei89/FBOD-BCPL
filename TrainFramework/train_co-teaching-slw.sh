#!/bin/bash
############### train CO-Teaching-SLW algorithm.##################
p_soft_label_func="piecewise"  ### under the mode SLW: linear or piecewise, otherwise not_applicable
p_Add_name="20240919"
p_Batch_size=8
modelB_total_train_val_loss_ap50_str=""
modelA_total_train_val_loss_ap50_str=""
log_txt="./logs_txt/log_"$p_Add_name".txt"

#### Shuffle train to two datasets #######################
python3 shuffle_train_to_two_datasets.py \
        --Add_name=$p_Add_name
#### Train the modelA 30 epoches with subsetAllA dataset under Normal mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=30 \
        --input_train_val_loss_ap50_str="None/None/None/0"

# update the object score of subsetAllB with modelA ###
python3 update_object_score.py \
        --data_subset="subsetAllB" \
        --modelAorB="modelA" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name
#############################################################################################################################
#############################################################################################################################
#### Train the modelB 30 epoches with subsetAllB dataset under SLW mode ##########################
modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllB" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=30 \
        --input_train_val_loss_ap50_str="None/None/None/0")
echo "modelB_total_train_val_loss_ap50_str: " >> $log_txt
echo $modelB_total_train_val_loss_ap50_str >> $log_txt
echo "" >> $log_txt
## update the object score of subsetAllA with modelB ###
python3 update_object_score.py \
        --data_subset="subsetAllA" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name

#### Train the modelA 30 epoches with subsetAllA dataset under SLW mode ##########################
modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=30 \
        --input_train_val_loss_ap50_str="None/None/None/0")
echo "modelA_total_train_val_loss_ap50_str: " >> $log_txt
echo $modelA_total_train_val_loss_ap50_str >> $log_txt
echo "" >> $log_txt
## update the object score of subsetAllB with modelA ###
python3 update_object_score.py \
        --data_subset="subsetAllB" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name
# #############################################################################################################################
# #############################################################################################################################

#### epoch 30 to epoch 80, every epoch update the score one time. #####
for n in {0..49};
do 
    s_epoch=$((30 + $n))
    e_epoch=$((30 + $n + 1))
    #### Train the modelB 1 epoch with subsetAllB dataset under SLW mode ##########################
    modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllB" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --input_train_val_loss_ap50_str="$modelB_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelB_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelB_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt

    ## update the object score of subsetAllA with modelB ###
    python3 update_object_score.py \
        --data_subset="subsetAllA" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name

    #### Train the modelA 1 epoch with subsetAllA dataset under SLW mode ##########################
    modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --input_train_val_loss_ap50_str="$modelA_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelA_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelA_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt

    ## update the object score of subsetAllB with modelA ###
    python3 update_object_score.py \
        --data_subset="subsetAllB" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name
    
done
#############################################################################################################################
#############################################################################################################################

#### Train the modelA 20 epoches with subsetAllA dataset under SLW mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllA" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=80 \
        --end_Epoch=100 \
        --input_train_val_loss_ap50_str="$modelA_total_train_val_loss_ap50_str"

#### Train the modelB 20 epoches with subsetAllB dataset under SLW mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAllB" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=80 \
        --end_Epoch=100 \
        --input_train_val_loss_ap50_str="$modelB_total_train_val_loss_ap50_str"
