import torch.nn as nn
import torch
from torchvision.ops import nms
import math
import copy

def box_iou(bbox1, bbox2):
    #### x1, y1, x2, y2
    inner_x1 = torch.max(bbox1[0], bbox2[0])
    inner_y1 = torch.max(bbox1[1], bbox2[1])
    inner_x2 = torch.min(bbox1[2], bbox2[2])
    inner_y2 = torch.min(bbox1[3], bbox2[3])
    inner_w = inner_x2 - inner_x1
    inner_h = inner_y2 - inner_y1
    if inner_w <= 0 or inner_h <= 0:
        return 0
    inner_area = (inner_w) * (inner_h)
    bbox1_area = (bbox1[2]-bbox1[0]) * ((bbox1[2]-bbox1[1]))
    bbox2_area = (bbox2[2]-bbox2[0]) * ((bbox2[2]-bbox2[1]))
    iou = inner_area/(bbox1_area + bbox2_area - inner_area)
    return iou

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(..., 4), xywh
    b2: tensor, shape=(..., 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(..., 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou

class FBObj():
    def __init__(self, score=None, image_id=False, bbox=None):
        self.score=score
        self.image_id = image_id
        self.bbox = bbox

class FB_boxdecoder(nn.Module):
    def __init__(self, model_input_size, score_threshold, nms_thres, scale=80.):
        super(FB_boxdecoder, self).__init__()
        self.model_input_size = model_input_size # h,w
        self.score_threshold = score_threshold
        self.nms_thres = nms_thres
        self.scale = scale

    def forward(self, input):
        # input is a list with with 2 members(CONF and LOC), each member is a 'bs,c,h,w' format tensor).
        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[0]/4 (stride = 4)
        in_w = input[0].size(3) # in_w

        # 2,bs,c,h,w -> 2,bs,h,w,c (a list with 2 members, each member is a 'bs,h,w,c' format tensor).

        # Branch for task, there are 2 tasks, that is CONF(Conf), and LOC(LOCation).
        # To get 3D tensor 'bs, h, w' or 4D tensor 'bs, h, w, c'.
        #################CONF
        # 
        predict_CONF = input[0] #bs,c,h,w  c=1
        predict_CONF = predict_CONF.view(bs,in_h,in_w) #bs,h,w
        ### bs, h, w
        predict_CONF = torch.sigmoid(predict_CONF)

        #################LOC
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # bs, h, w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[1]/in_w) + (self.model_input_size[1]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, w, h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[0]/in_h) + (self.model_input_size[0]/in_h)/2).repeat(bs, 1, 1)
        # bs, w, h -> bs, h, w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_LOC = input[1] #bs, c,h,w  c=4(dx1,dy1,dx2,dy2)
         # bs, c, h, w -> bs, h, w, c
        predict_LOC = predict_LOC.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale + ref_point_ys
        # print("predict_CONF")
        # print(predict_CONF)

        CONF_mask = predict_CONF > self.score_threshold
        # CONF_mask_test = predict_CONF > 0.0014
        # print("predict_CONF[0][CONF_mask_test[0]]")
        # print(predict_CONF[0][CONF_mask_test[0]])

        outputs = []
        for b in range(bs):
            ### For one batch
            ### 1 dim tensor  [n1] n1 is the numper of the obj.
            predict_CONF_b = predict_CONF[b][CONF_mask[b]]
            ### 2 dim tensor  [n1, 1] n1 is the numper of the obj.
            predict_CONF_b = predict_CONF_b.unsqueeze(1)

            ### 2 dim tensor  [n1, 4] n1 is the numper of the obj.
            predict_LOC_b = predict_LOC[b][CONF_mask[b]]

            ### 2 dim tensor  [n3, 5] n3 is the numper of the obj.
            detections = torch.concat([predict_LOC_b, predict_CONF_b], dim=1)

            keep = nms(detections[:, :4], detections[:, 4], self.nms_thres)
            max_detections = detections[keep]
            outputs.append(max_detections)
        return outputs

def Soft_Label_Remap(image_info_list, Accum_Count_HElevel):
    
    P_Upper_Lower = [[0,0],[0,0],[0,0],[0,0]] ### The Upper and Lower value of the Post confidence for four HElevels
    S_Upper_Lower = [[1, 0.75],[0.75, 0.5],[0.5, 0.25],[0.25, 0]] ### The Upper and Lower object score for four HElevels

    post_conf_list = []
    for image_info_instance in image_info_list:
        for box_info_instance in image_info_instance.box_info_list:
            post_conf_list.append(box_info_instance.post_score)
    post_conf_list.sort(reverse=True)

    for i in range(4):
        if i == 0 :
            P_Upper_Lower[i][0] = post_conf_list[0]
        else:
            P_Upper_Lower[i][0] = post_conf_list[Accum_Count_HElevel[i-1]]
        P_Upper_Lower[i][1] = post_conf_list[Accum_Count_HElevel[i]-1]

    P_Max_Min = [[0,0],[0,0],[0,0],[0,0]] ### The Max and Min value of the Post confidence for four HElevels
    S_Max_Min = [[0,0],[0,0],[0,0],[0,0]] ### The Max and Min object score for four HElevels
    #### To determin the P_Max_Min and S_Max_Min through regulation
    for i in range(4):
        if (P_Upper_Lower[i][0] <= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] <= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = S_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin

        elif (P_Upper_Lower[i][0] <= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] >= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = P_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = P_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = P_Upper_Lower[i][1] ### Smin

        elif (P_Upper_Lower[i][0] >= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] <= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = P_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = P_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin
        
        elif (P_Upper_Lower[i][0] >= S_Upper_Lower[i][0]) and (P_Upper_Lower[i][1] >= S_Upper_Lower[i][1]):
            P_Max_Min[i][0] = P_Upper_Lower[i][0] ### Pmax
            S_Max_Min[i][0] = S_Upper_Lower[i][0] ### Smax

            P_Max_Min[i][1] = S_Upper_Lower[i][1] ### Pmin
            S_Max_Min[i][1] = S_Upper_Lower[i][1] ### Smin
    
    ### 
    for image_info_instance in image_info_list:
        for box_info_instance in image_info_instance.box_info_list:
            p = box_info_instance.post_score
            if p >= P_Upper_Lower[0][1]:
                box_info_instance.post_HElevel = 0
            elif p >= P_Upper_Lower[1][1]:
                box_info_instance.post_HElevel = 1
            elif p >= P_Upper_Lower[2][1]:
                box_info_instance.post_HElevel = 2
            else:
                box_info_instance.post_HElevel = 3
            

            index = box_info_instance.post_HElevel
            p_max = P_Max_Min[index][0]
            p_min = P_Max_Min[index][1]
            s_max = S_Max_Min[index][0]
            s_min = S_Max_Min[index][1]

            if p == p_min:
                box_info_instance.post_score = s_min
            else:
                box_info_instance.post_score = (s_max*(p-p_min) + s_min*(p_max-p))/(p_max-p_min)

def Soft_Label_Remap2(image_info_list, Count_HElevel):
    post_conf_list = []
    for image_info_instance in image_info_list:
        for box_info_instance in image_info_instance.box_info_list:
            post_conf_list.append(box_info_instance.post_score)
    post_conf_list.sort(reverse=True)

    min_Easy_score = post_conf_list[Count_HElevel[0]]
    max_Hard_score = post_conf_list[Count_HElevel[0]+1]

    if min_Easy_score >= 0.25 and max_Hard_score <= 0.25:
        return 0
    elif min_Easy_score > 0.25 and max_Hard_score >= 0.25:
        for image_info_instance in image_info_list:
            for box_info_instance in image_info_instance.box_info_list:
                p = box_info_instance.post_score
                if p <= max_Hard_score:
                    box_info_instance.post_HElevel = 1
                    box_info_instance.post_score = p*(0.25/max_Hard_score)
                else:
                    box_info_instance.post_HElevel = 0

    elif min_Easy_score < 0.25:
        for image_info_instance in image_info_list:
            for box_info_instance in image_info_instance.box_info_list:
                p = box_info_instance.post_score
                if p >= min_Easy_score:
                    box_info_instance.post_HElevel = 0
                    box_info_instance.post_score = (1*(p-min_Easy_score) + 0.25*(1-p))/(1-min_Easy_score)
                else:
                    box_info_instance.post_HElevel = 1