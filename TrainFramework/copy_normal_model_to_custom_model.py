from config.opts import opts
import os
import shutil

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
    
    normal_model_A_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + abbr_assign_method \
                             + "_modelA/"
    
    normal_model_B_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_Normal_" + abbr_assign_method \
                             + "_modelB/"

    custom_model_A_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelA/"
    
    custom_model_B_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelB/"
    
    os.makedirs(custom_model_A_dir, exist_ok=True)
    os.makedirs(custom_model_B_dir, exist_ok=True)
    model_name = "FB_object_detect_model.pth"

    shutil.copy(normal_model_A_dir + model_name, custom_model_A_dir + model_name)
    shutil.copy(normal_model_B_dir + model_name, custom_model_B_dir + model_name)