import cv2
import os
from FB_detector import FB_detector
from utils.common import load_data_raw_resize_boxes, GetMiddleImg_ModelInput_for_MatImageList
from config.opts import opts
import numpy as np
import shutil
from tqdm import tqdm

from utils.get_box_info_list import getBoxInfoListForOneImage, image_info


os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

if __name__ == "__main__":
    opt = opts().parse()
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W

    input_img_num = opt.input_img_num
    aggregation_output_channels = opt.aggregation_output_channels
    aggregation_method = opt.aggregation_method
    input_mode=opt.input_mode
    backbone_name = opt.backbone_name
    fusion_method = opt.fusion_method
    learn_mode = opt.learn_mode
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "binary_assign":
        abbr_assign_method = "ba"
    elif opt.assign_method == "guassian_assign":
        abbr_assign_method = "ga"
    elif opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! abbr_assign_method error.")
    base_Add_name = opt.Add_name
    if opt.learn_mode == "CPLBC":
        Add_name = opt.MF_para + "_"  + opt.TS_para + "_" + opt.Add_name
    elif opt.learn_mode == "CPL":
        Add_name = opt.cpl_mode + "_" + opt.Add_name
    else:
        Add_name = opt.Add_name
    model_name="FB_object_detect_model.pth"

    data_subset = opt.data_subset ### subsetA, subsetB or subsetAll.
    modelAorB = opt.modelAorB  #### modelA or modelB

    # FB_detector parameters
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.
    
    fb_detector = FB_detector(model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              learn_mode=learn_mode, abbr_assign_method=abbr_assign_method, Add_name=Add_name + "_"  + modelAorB, model_name=model_name)
    Cuda = True
    annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + base_Add_name + "_" + data_subset + ".txt"
    if os.path.exists(annotation_path):
        pass
    else:
        # source_annotation_path = "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
        # os.makedirs("./variable_score/", exist_ok=True)
        # shutil.copy(source_annotation_path, annotation_path)
        raise ValueError(f"Error! No train_annotation_path: {annotation_path}")
    dataset_image_path = opt.data_root_path + "images/train/"

    get_box_info_for_one_image = getBoxInfoListForOneImage(image_size = (model_input_size[1],model_input_size[0])) # image_size w,h

    
    with open(annotation_path) as f:
        lines = f.readlines()
    
    image_info_list = []
    with tqdm(total=len(lines)) as pbar:
        for line in lines:
            images, raw_bboxes, bboxes, first_img_name = load_data_raw_resize_boxes(line, dataset_image_path, frame_num=opt.input_img_num, image_size=model_input_size)
            image_info_instance = image_info(iname=first_img_name)
            if len(bboxes) == 0:
                image_info_list.append(image_info_instance)
                pbar.update(1)
                continue
            _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=opt.input_img_num, input_mode=opt.input_mode)
            predictions = fb_detector.inference(model_input)
            image_info_instance.box_info_list = get_box_info_for_one_image(predictions, raw_bboxes, bboxes)
            image_info_list.append(image_info_instance)
            pbar.update(1)

    ### Update the object score of annotation by rewriting all the information.
    annotation_file = open(annotation_path,'w')
    for image_info_instance in image_info_list:
        annotation_file.write(image_info_instance.iname)
        if len(image_info_instance.box_info_list) == 0:
            annotation_file.write(" None")
        else:
            for box_info_instance in image_info_instance.box_info_list:
                # Use the box_info_instance.post_score to update difficulty
                string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(box_info_instance.post_score)
                annotation_file.write(string_label)
        annotation_file.write("\n")
    annotation_file.close()

