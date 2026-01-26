import torch
from collections import Counter

from FB_detector import FB_detector
import numpy as np
from utils.common import load_data, GetMiddleImg_ModelInput_for_MatImageList
from utils.utils import FBObj
from config.opts import opts



def IOU(box1, box2):
    """
        计算IOU
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def mean_average_precision(pred_oriented_objs,true_oriented_objs,iou_threshold):

    epsilon=1e-6# Prevent the denominator from being 0
    detections = pred_oriented_objs
    ground_truths = true_oriented_objs
    # img 0 has 3 bboxes
    # img 1 has 5 bboxes
    # Just like this: amount_bboxes={0:3,1:5}
    # The number of true boxes in each image is counted, and train_idx specifies the image number to distinguish between each image
    amount_bboxes=Counter(gt.image_id for gt in ground_truths)

    for key,val in amount_bboxes.items():
        amount_bboxes[key]=torch.zeros(val)#Set to 0, this means that none of the real boxes initially matched any of the predicted boxes

    # Sort the prediction boxes in descending order of confidence
    detections.sort(key=lambda x:x.score,reverse=True)

    # Initialize TP and FP.
    TP=torch.zeros(len(detections))
    FP=torch.zeros(len(detections))

        # TP+FN is just the total number of GT boxes of the current category, which is fixed
    total_true_bboxes=len(ground_truths)
    
    # If none of the GT boxes are available, then we simply return
    if total_true_bboxes == 0:
        return 0, 0, 0

    # For each prediction box, we first find all the real boxes in the image it is in, and then calculate the 
    # IoU between the prediction box and each real box. If the IoU is greater than the IoU threshold and the 
    # real box does not match other prediction boxes, we set the prediction result of the prediction box as TP. 
    # Otherwise FP for detection_idx,detection in enumerate(detections)
    for detection_idx,detection in enumerate(detections):
        # IoU can be calculated only for boxes within the same image, not between different images.
        # There is a zeroth dimension to the numbering of images
        # So the following code finds all the actual boxes in the image where the detection is predicted, and uses that to calculate the IoU
        ground_truth_img=[bbox for bbox in ground_truths if bbox.image_id==detection.image_id]

        best_iou=0
        for idx,gt in enumerate(ground_truth_img):
            # Calculate the IoU between the predicted detection and every real box in the image it is in
            iou=IOU(detection.bbox,gt.bbox)
            if iou >best_iou:
                best_iou=iou
                best_gt_idx=idx
        if best_iou>iou_threshold:
            # Here, detection[0] is a key of amount_bboxes, representing the image number, and best_gt_idx is the index of the ground truth box corresponding to this key's value.
            if amount_bboxes[detection.image_id][best_gt_idx]==0:# Only real boxes that are not occupied can be used, with 0 indicating that they are not occupied (occupied: the real box matches the predicted box [IoU is greater than the set IoU threshold]).
                TP[detection_idx]=1# This prediction box is TP
                amount_bboxes[detection.image_id][best_gt_idx]=1# Mark the real box as used and not used for other prediction boxes. Because a predicted box can correspond to at most one true box (at most: if the IoU is less than the IoU threshold, the predicted box has no corresponding true box)
            else:
                FP[detection_idx]=1# Although the IoU between this predicted box and a box in the real box is greater than the IoU threshold, this real box has been matched with other predicted boxes, so the predicted box is FP
        else:
            FP[detection_idx]=1# The IoU between this predicted box and every box in the true box is less than the IoU threshold, so the predicted box is directly FP
    TP_cumsum=torch.cumsum(TP,dim=0)
    FP_cumsum=torch.cumsum(FP,dim=0)

    TP_sum = torch.sum(TP)
    FP_sum = torch.sum(FP)
    
    # Apply the formula for calculation.
    recalls=TP_cumsum/(total_true_bboxes+epsilon)
    precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))

    recalls_value=TP_sum/(total_true_bboxes+epsilon)
    precisions_value=TP_sum/(TP_sum+FP_sum+epsilon)

    # Add the point [0,1] to it
    precisions=torch.cat((torch.tensor([1]),precisions))
    recalls=torch.cat((torch.tensor([0]),recalls))
    # Use trapz to calculate AP
    average_precision_value = torch.trapz(precisions,recalls)

    return average_precision_value, recalls_value, precisions_value

def labels_to_results(bboxes, image_id):
    label_obj_list = []
    for bbox in bboxes:
        label_obj_list.append(FBObj(score=1.0, image_id=image_id, bbox=bbox[:4]))
    return label_obj_list

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
    
    if opt.prior_way == "ASP":
        Add_name = opt.prior_way + "_" + opt.Add_name
        prior_learn_mode = "All_Sample"
    elif opt.prior_way == "ESP":
        Add_name = opt.prior_way + "_" + opt.Add_name
        prior_learn_mode = "Easy_Sample"
    else:
        Add_name = opt.Add_name
        
    if opt.learn_mode == "CPLBC":
        Add_name = opt.MF_para + "_"  + opt.TS_para + "_" + Add_name
    elif opt.learn_mode == "CPL":
        Add_name = opt.cpl_mode + "_" + Add_name
    else:
        Add_name = Add_name
    Add_name = Add_name + "_" + opt.modelAorB
    
    model_name=opt.model_name

    # FB_detector parameters
    # model_input_size=(384,672),
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.
    
    fb_detector = FB_detector(model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              learn_mode=learn_mode, abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name)
    
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty.txt"
    dataset_image_path = "../images/"

    # val_lines = open(annotation_path).readlines()
    # # # 0.1 is used for validation and 0.9 for training
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split) 
    num_train = len(lines) - num_val

    train_lines = lines[:num_train]
    val_lines = lines[num_train:]

    all_label_obj_list = []
    all_obj_result_list = []
    for i, line in enumerate(val_lines):
        images, bboxes, _ = load_data(line, dataset_image_path, frame_num=input_img_num)
        raw_image_shape = np.array(images[0].shape[0:2]) # h,w
        all_label_obj_list += labels_to_results(bboxes, i)

        _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        outputs = fb_detector.detect_image(model_input, raw_image_shape=raw_image_shape)

        if outputs != None:
            obj_result_list = []
            for output in outputs:
                box = output[:4]
                score = output[4]
                obj_result_list.append(FBObj(score=score, image_id=i, bbox=box))
            all_obj_result_list += obj_result_list
    AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5) # Include background
    print("AP_50,REC_50,PRE_50:")
    print(AP_50,REC_50,PRE_50)
    AP_75,REC_75,PRE_75=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.75)
    print("AP_75,REC_75,PRE_75:")
    print(AP_75,REC_75,PRE_75)
    mAP = 0
    for i in range(50,95,5):
        iou_t = i/100
        mAP_, _, _ = mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=iou_t)
        mAP += mAP_
    mAP = mAP/10
    print("mAP = ",mAP)