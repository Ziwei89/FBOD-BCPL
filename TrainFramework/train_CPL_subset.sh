#!/bin/bash
############### train CPL_subset algorithm.##################
p_spl_mode="soft"  ### under the mode splbc: hard or soft
p_total_Epoch=100
p_Add_name="20241012"
p_Batch_size=8
modelB_total_train_val_loss_ap50_str=None/None/None/0
modelA_total_train_val_loss_ap50_str=None/None/None/0
log_txt="./logs_txt/log_"$p_Add_name".txt"
# ############### Random split train to two sub datasets ##################
# python3 random_split_train_to_two_sub_datasets.py \
#         --Add_name=$p_Add_name
# #### Train the modelA 30 epoches with subsetA dataset under Normal mode ##########################
# python3 train_AP50.py \
#         --data_augmentation \
#         --data_subset="subsetA" \
#         --modelAorB="modelA" \
#         --learn_mode="Normal" \
#         --Add_name=$p_Add_name \
#         --Batch_size=$p_Batch_size \
#         --start_Epoch=0 \
#         --end_Epoch=30 \
#         --input_train_val_loss_ap50_str="None/None/None/0"
# update the object score of subsetB with modelA ###
python3 update_object_score.py \
        --data_subset="subsetB" \
        --modelAorB="modelA" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name

# #### Train the modelB 30 epoches with subsetB dataset under Normal mode ##########################
# python3 train_AP50.py \
#         --data_augmentation \
#         --data_subset="subsetB" \
#         --modelAorB="modelB" \
#         --learn_mode="Normal" \
#         --Add_name=$p_Add_name \
#         --Batch_size=$p_Batch_size \
#         --start_Epoch=0 \
#         --end_Epoch=30 \
#         --input_train_val_loss_ap50_str="None/None/None/0"
# update the object score of subsetA with modelB ###
python3 update_object_score.py \
        --data_subset="subsetA" \
        --modelAorB="modelB" \
        --learn_mode="Normal" \
        --Add_name=$p_Add_name

### Rename folder and pic name from Normal learn mode #######

python3 rename_folder_name_pic_name_from_normal.py \
        --learn_mode="SPLBC" \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name

#############################################################################################################################
#############################################################################################################################
#### epoch 0 to epoch 80, every epoch update the score one time. #####
for n in {0..79};
do 
    s_epoch=$(($n))
    e_epoch=$(($n + 1))

    #### Train the modelA 1 epoch with subsetA dataset under SPLBC mode ##########################
    modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetA" \
        --modelAorB="modelA" \
        --learn_mode="SPLBC" \
        --load_pretrain_model \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --Batch_size=$p_Batch_size \
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
        --learn_mode="SPLBC" \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name

    #### Train the modelB 1 epoch with subsetB dataset under SPLBC mode ##########################
    modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetB" \
        --modelAorB="modelB" \
        --learn_mode="SPLBC" \
        --load_pretrain_model \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --Batch_size=$p_Batch_size \
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
        --learn_mode="SPLBC" \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name
done
#############################################################################################################################
#############################################################################################################################

## Merge the two sub dataset
python3 Merge_two_sub_datasets.py \
        --Add_name=$p_Add_name


#############################################################################################################################
#############################################################################################################################
#### epoch 80 to epoch 100, every epoch update the score one time. #####
for n in {80..99};
do 
    s_epoch=$(($n))
    e_epoch=$(($n + 1))

    #### Train the modelA 1 epoch with subsetAll dataset under SPLBC mode ##########################
    modelA_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAll" \
        --modelAorB="modelA" \
        --learn_mode="SPLBC" \
        --load_pretrain_model \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --Batch_size=$p_Batch_size \
        --input_train_val_loss_ap50_str="$modelA_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelA_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelA_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt

    ## update the object score of subsetAll with modelA ###
    python3 update_object_score.py \
        --data_subset="subsetAll" \
        --modelAorB="modelA" \
        --learn_mode="SPLBC" \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name

    #### Train the modelB 1 epoch with subsetAll dataset under SPLBC mode ##########################
    modelB_total_train_val_loss_ap50_str=$(python3 train_AP50.py \
        --data_augmentation \
        --data_subset="subsetAll" \
        --modelAorB="modelB" \
        --learn_mode="SPLBC" \
        --load_pretrain_model \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name \
        --Batch_size=$p_Batch_size \
        --start_Epoch=$s_epoch \
        --end_Epoch=$e_epoch \
        --total_Epoch=$p_total_Epoch \
        --Batch_size=$p_Batch_size \
        --input_train_val_loss_ap50_str="$modelB_total_train_val_loss_ap50_str")
    
    echo "Epoch: " >> $log_txt
    echo $e_epoch >> $log_txt
    echo "modelB_total_train_val_loss_ap50_str: " >> $log_txt
    echo $modelB_total_train_val_loss_ap50_str >> $log_txt
    echo "" >> $log_txt

    ## update the object score of subsetAll with modelB ###
    python3 update_object_score.py \
        --data_subset="subsetAll" \
        --modelAorB="modelB" \
        --learn_mode="SPLBC" \
        --spl_mode=$p_spl_mode \
        --Add_name=$p_Add_name
done
#############################################################################################################################
#############################################################################################################################