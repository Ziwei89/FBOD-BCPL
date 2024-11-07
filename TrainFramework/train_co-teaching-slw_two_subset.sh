#!/bin/bash
############### train CO-Teaching-SLW algorithm.##################
p_soft_label_func="linear"  ### under the mode SLW: linear or piecewise, otherwise not_applicable
p_Add_name="20240911"
p_Batch_size=8
modelB_total_train_val_loss_ap50_str=""
modelA_total_train_val_loss_ap50_str=""
log_txt="./logs_txt/log_"$p_Add_name".txt"
############### Random split train to two sub datasets ##################
python3 random_split_train_to_two_sub_datasets.py \
        --Add_name=$p_Add_name
#### Train the modelA 30 epoches with subsetA dataset under Normal mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetA" \
        --modelAorB="modelA" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=0 \
        --end_Epoch=30 \
        --input_train_val_loss_ap50_str="None/None/None/0"

# update the object score of subsetB with modelA ###
python3 update_object_score.py \
        --data_subset="subsetB" \
        --modelAorB="modelA" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name
#############################################################################################################################
#############################################################################################################################
#### Train the modelB 30 epoches with subsetB dataset under SLW mode ##########################
modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetB" \
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
## update the object score of subsetA with modelB ###
python3 update_object_score.py \
        --data_subset="subsetA" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name

#### Train the modelA 30 epoches with subsetA dataset under SLW mode ##########################
modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetA" \
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
## update the object score of subsetB with modelA ###
python3 update_object_score.py \
        --data_subset="subsetB" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name
#############################################################################################################################
#############################################################################################################################
#### epoch 30 to epoch 80, every 10 epoches update the score one time. #####
for n in {0..4};
do 
    s_epoch=$((30 + $n * 10))
    e_epoch=$((30 + $n * 10 + 10))
    #### Train the modelB 10 epoches with subsetB dataset under SLW mode ##########################
    modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetB" \
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

    ## update the object score of subsetA with modelB ###
    python3 update_object_score.py \
        --data_subset="subsetA" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name

    #### Train the modelA 10 epoches with subsetA dataset under SLW mode ##########################
    modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetA" \
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

    ## update the object score of subsetB with modelA ###
    python3 update_object_score.py \
        --data_subset="subsetB" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name
    
done
#############################################################################################################################
#############################################################################################################################

## Merge the two sub dataset
python3 Merge_two_sub_datasets.py \
        --Add_name=$p_Add_name \
        --soft_label_func=$p_soft_label_func

#### Train the modelA 70 epoches with subsetAll dataset under SLW mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAll" \
        --modelAorB="modelA" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=80 \
        --end_Epoch=150 \
        --input_train_val_loss_ap50_str="$modelA_total_train_val_loss_ap50_str"

#### Train the modelB 70 epoches with subsetAll dataset under SLW mode ##########################
python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAll" \
        --modelAorB="modelB" \
        --learn_mode="SLW" \
        --soft_label_func=$p_soft_label_func \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=80 \
        --end_Epoch=150 \
        --input_train_val_loss_ap50_str="$modelB_total_train_val_loss_ap50_str"
