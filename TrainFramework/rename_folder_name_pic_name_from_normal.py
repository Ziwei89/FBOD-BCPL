import os
from config.opts import opts

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}


if __name__ == "__main__":

    opt = opts().parse()
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! assign_method error.")
    
    if opt.learn_mode == "SLW" and opt.soft_label_func == "not_applicable":
        raise("Error! when opt.learn_mode = 'SLW', the opt.soft_label_func cannot equal to 'not_applicable'.")
    
    base_Add_name = opt.Add_name
    if opt.learn_mode == "SLW":
        Add_name = opt.soft_label_func + "_" + opt.Add_name
    elif opt.learn_mode == "SPLBC":
        Add_name = opt.spl_mode + "_" + opt.Add_name
    else:
        Add_name = opt.Add_name


    
    raw_save_model_dir_modelA = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + abbr_assign_method \
                             + "_"  + base_Add_name + "_modelA/"
    
    save_model_dir_modelA = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelA/"
    
    os.rename(raw_save_model_dir_modelA, save_model_dir_modelA)
    print("rename: ", raw_save_model_dir_modelA, " to: ", save_model_dir_modelA)

    
    raw_save_model_dir_modelB = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + abbr_assign_method \
                             + "_"  + base_Add_name + "_modelB/"
    
    save_model_dir_modelB = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelB/"
    
    os.rename(raw_save_model_dir_modelB, save_model_dir_modelB)
    print("rename: ", raw_save_model_dir_modelB, " to: ", save_model_dir_modelB)

    ############### For log figure ################
    raw_log_pic_name_loss_modelA = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + \
                                                abbr_assign_method + "_" + base_Add_name + "_modelA_loss.jpg"
    
    log_pic_name_loss_modelA = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode \
                                            + "_" + abbr_assign_method + "_" + Add_name + "_modelA_loss.jpg"
    
    os.rename(raw_log_pic_name_loss_modelA, log_pic_name_loss_modelA)
    print("rename: ", raw_log_pic_name_loss_modelA, " to: ", log_pic_name_loss_modelA)
    
    
    raw_log_pic_name_loss_modelB = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + \
                                                abbr_assign_method + "_" + base_Add_name + "_modelB_loss.jpg"
    
    log_pic_name_loss_modelB = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode \
                                            + "_" + abbr_assign_method + "_" + Add_name + "_modelB_loss.jpg"
    
    os.rename(raw_log_pic_name_loss_modelB, log_pic_name_loss_modelB)
    print("rename: ", raw_log_pic_name_loss_modelB, " to: ", log_pic_name_loss_modelB)
    
    
    raw_log_pic_name_ap50_modelA = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + \
                                                abbr_assign_method + "_" + base_Add_name + "_modelA_ap50.jpg"
    
    log_pic_name_ap50_modelA = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode \
                                            + "_" + abbr_assign_method + "_" + Add_name + "_modelA_ap50.jpg"
    
    os.rename(raw_log_pic_name_ap50_modelA, log_pic_name_ap50_modelA)
    print("rename: ", raw_log_pic_name_ap50_modelA, " to: ", log_pic_name_ap50_modelA)
    

    raw_log_pic_name_ap50_modelB = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + \
                                                abbr_assign_method + "_" + base_Add_name + "_modelB_ap50.jpg"
    
    log_pic_name_ap50_modelB = "train_output_img/" + num_to_english_c_dic[opt.input_img_num] + "/" +opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                            + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode \
                                            + "_" + abbr_assign_method + "_" + Add_name + "_modelB_ap50.jpg"
    
    os.rename(raw_log_pic_name_ap50_modelB, log_pic_name_ap50_modelB)
    print("rename: ", raw_log_pic_name_ap50_modelB, " to: ", log_pic_name_ap50_modelB)
    ################################################