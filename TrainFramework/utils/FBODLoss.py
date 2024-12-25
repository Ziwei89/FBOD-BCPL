import torch
import torch.nn as nn
import sys
import math
import numpy as np
sys.path.append("..")
from .getDynamicTargets import getTargets

def MSELoss(pred,target):
    return (pred-target)**2

def BCELoss(pred,target):
    criterion = torch.nn.BCELoss(reduction='none')
    return criterion(pred, target)

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, 1)
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

class LossFunc(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale=80., stride=2, learn_mode="CPLBC", MF_para=1.0/3, cuda=True, gettargets=False):
        super(LossFunc, self).__init__()
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.scale = scale
        self.learn_mode = learn_mode
        self.MF_para = MF_para #### Minimizer Function parameter
        #(model_input_size, num_classes=2, stride=2)
        self.get_targets = getTargets(model_input_size, num_classes, scale=scale, stride=stride, cuda=True)
        self.cuda = cuda
        self.gettargets = gettargets
    
    def forward(self, input, targets, cpl_threshold=None):

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # targets is bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        if self.gettargets:
            if self.learn_mode == "All_Sample":
                targets = self.get_targets(input, targets, difficult_mode=0) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "Easy_Sample":
                targets = self.get_targets(input, targets, difficult_mode=1) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "CPLBC":
                targets = self.get_targets(input, targets, difficult_mode=2, cpl_threshold=cpl_threshold, MF_para=self.MF_para)
            elif self.learn_mode == "CPL" or self.learn_mode == "HEM":
                targets = self.get_targets(input, targets, difficult_mode=3)
            else:
                raise("Error! learn_mode error.")

        # input is a list with with 2 members(CONF and LOC), each member is a 'bs,c,in_h,in_w' format tensor).
        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/stride (stride = 2)
        in_w = input[0].size(3) # in_w

        # 2,bs,c,in_h,in_w -> 2,bs,in_h,in_w,c (a list with 2 members, each member is a 'bs,in_h,in_w,c' format tensor).

        # Branch for task, there are 2 tasks, that is CONF(CONFidence), and LOC(LOCation).
        # To get 3D tensor 'bs, in_h, in_w' or 4D tensor 'bs, in_h, in_w, c'.
        #################CONF
        # 
        predict_CONF = input[0].type(FloatTensor) #bs,c,in_h,in_w  c=1
        predict_CONF = predict_CONF.view(bs,in_h,in_w) #bs,in_h,in_w
        ### bs, in_h, in_w
        predict_CONF = torch.sigmoid(predict_CONF)


        #################LOC

        # bs, in_h, in_w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, in_w, in_h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, in_w, in_h -> bs, in_h, in_w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_LOC = input[1].type(FloatTensor) #bs, c,in_h,in_w  c=4(dx1,dy1,dx2,dy2)
        # bs, c, in_h, in_w -> bs, in_h, in_w, c
        predict_LOC = predict_LOC.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale + ref_point_ys

        ### bs, in_h, in_w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_LOC[..., 2] = predict_LOC[..., 2] - predict_LOC[..., 0]
        predict_LOC[..., 3] = predict_LOC[..., 3] - predict_LOC[..., 1]
        predict_LOC[..., 0] = predict_LOC[..., 0] + predict_LOC[..., 2]/2
        predict_LOC[..., 1] = predict_LOC[..., 1] + predict_LOC[..., 3]/2
        ###########################

        # targets is a list wiht 2 members, each is a 'bs*in_h,in_w*c' format tensor(cls and bbox).
        # 2,bs*c*in_h,in_w -> 3,bs,in_h,in_w,c (a list with 3 members, each member is a 'bs,in_h,in_w,c' format tensor).

        #################CONF_CLS
        ### bs, in_h, in_w, c(c=num_classes(Include background))
        label_CONF_CLS = targets[0].type(FloatTensor) #bs*in_h,in_w*c  c=num_classes(Include background)
        label_CONF_CLS = label_CONF_CLS.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c
        ### bs, in_h, in_w
        label_CONF = torch.sum(label_CONF_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Guassian Heat Conf
        label_CLS_weight =  torch.ceil(label_CONF_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_neg = weight_neg.squeeze(3)
        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)
        ### bs
        bs_neg_nums = torch.sum(weight_neg, dim=(1,2))
        ### bs
        bs_obj_nums = torch.sum(weight_pos, dim=(1,2))
        
        #################LOC
        label_LOC_sampleweight_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_sampleweight_lamda = label_LOC_sampleweight_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_sampleweight_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
        ### bs, in_h, in_w
        label_sampleweight = label_LOC_sampleweight_lamda[:,:,:,4] # bs,in_h,in_w
        ### bs, in_h, in_w
        # label_lamda = label_LOC_sampleweight_lamda[:,:,:,5] # bs,in_h,in_w

        ## Conf Loss
        ## bs, in_h, in_w
        # print("predict_CONF[predict_CONF>0.2]")
        # print(predict_CONF[predict_CONF>0.2])
        MSE_Loss = MSELoss(label_CONF, predict_CONF)
        neg_MSE_Loss = MSE_Loss * weight_neg
        pos_MSE_Loss = (MSE_Loss * label_sampleweight) * weight_pos

        CONF_loss = 0
        for b in range(bs):
            CONF_loss_per_batch = 0
            ### in_h, in_w
            if bs_obj_nums[b] != 0:
                k = bs_obj_nums[b].cpu()
                k = int(k.numpy())
                topk = 2*k
                if topk > bs_neg_nums[b]:
                    topk = bs_neg_nums[b]
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk((neg_MSE_Loss[b]).view(-1), topk).values)
                pos_MSE_Loss_sum = torch.sum(pos_MSE_Loss[b])
                CONF_loss_per_batch = (neg_MSE_Loss_topk_sum + 10*pos_MSE_Loss_sum)/bs_obj_nums[b]
            else:
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk(neg_MSE_Loss[b], 20).values)
                CONF_loss_per_batch = neg_MSE_Loss_topk_sum/10
            CONF_loss += CONF_loss_per_batch
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        ciou_loss = (ciou_loss.view(bs,in_h,in_w)) * label_sampleweight * weight_pos
        LOC_loss = 0
        for b in range(bs):
            LOC_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                LOC_loss_per_batch = torch.sum(ciou_loss[b])/bs_obj_nums[b]
            else:
                LOC_loss_per_batch = 0
            LOC_loss += LOC_loss_per_batch

        total_loss = (10*CONF_loss + 100*LOC_loss) / bs
        return total_loss

class Box_info_for_difficulty_loss(object):
    def __init__(self, box_id, bbox):
        ### bbox: bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] object score (difficult)
        self.__box_id = box_id
        self.__label_difficulty = bbox[5]
        self.__positive_points_map = None

    @property
    def box_id(self):
        return self.__box_id
    @property
    def label_difficulty(self):
        return self.__label_difficulty
    @property
    def positive_points_map(self):
        temp_positive_points_map = self.__positive_points_map
        self.__positive_points_map=None
        return temp_positive_points_map
    
    @positive_points_map.setter
    def positive_points_map(self,positive_points_map):
        self.__positive_points_map=positive_points_map

def min_max_ref_point_index(bbox, output_feature, image_size):
    min_x = bbox[0] - bbox[2]/2
    min_y = bbox[1] - bbox[3]/2

    max_x = bbox[0] + bbox[2]/2
    max_y = bbox[1] + bbox[3]/2
    min_wight_index = math.floor(max((min_x*output_feature[0])/image_size[0] - 1/2,0))
    min_height_index = math.floor(max((min_y*output_feature[1])/image_size[1] - 1/2,0))

    max_wight_index = math.ceil(min((max_x*output_feature[0])/image_size[0] - 1/2,output_feature[0]-1))
    max_height_index = math.ceil(min((max_y*output_feature[1])/image_size[1] - 1/2,output_feature[1]-1))

    return (min_wight_index, min_height_index, max_wight_index, max_height_index)

def is_point_in_bbox(point, bbox):
    condition1 = (point[0] >= bbox[0]-bbox[2]/2) and (point[0] <= bbox[0]+bbox[2]/2)
    condition2 = (point[1] >= bbox[1]-bbox[3]/2) and (point[1] <= bbox[1]+bbox[3]/2)
    if condition1 and condition2:
        return True
    else:
        return False

def difficulty_loss_per_batch(difficulty_map, bboxes, out_feature_size, image_size, FloatTensor):
    
    ## difficulty_map (in_h, in_w) tensor
    ## bboxes (n,6) tensor (6: x1, y1, x2, y2, class_id, object score)
    # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

    box_id_list = []
    box_info_list = []
    box_ids_map = np.array([-1.]*int(out_feature_size[0])*int(out_feature_size[1]))
    sample_position_list = []

    for box_id, bbox in enumerate(bboxes):
        obj_area = bbox[2] * bbox[3]
        if obj_area == 0:
            continue
        box_id_list.append(box_id)
        ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
        box_info_list.append(Box_info_for_difficulty_loss(box_id=box_id, bbox=bbox))

        min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,out_feature_size,image_size)
        for i in range(min_height_index, max_height_index+1):
            for j in range(min_wight_index, max_wight_index+1):
                ref_point_position = []
                ref_point_position.append(j*(image_size[0]/out_feature_size[0]) + (image_size[0]/out_feature_size[0])/2) #### x
                ref_point_position.append(i*(image_size[1]/out_feature_size[1]) + (image_size[1]/out_feature_size[1])/2) #### y

                if is_point_in_bbox(ref_point_position, bbox[:4]):# The point is in bbox.
                    if (i,j) in sample_position_list:
                        box_ids_map[i*int(out_feature_size[0]) + j] = -1 # # Ignore this point
                    else:
                        sample_position_list.append((i,j))
                        box_ids_map[i*int(out_feature_size[0]) + j] = box_id # box_id

    for box_id in box_id_list:
        True_False_map = box_ids_map == box_id
        if any(True_False_map): # If any member of box_id_map is True, return True; all the member is False, return False.
            for box_info in box_info_list:
                if box_info.box_id == box_id:
                    True_False_map = True_False_map.reshape(int(out_feature_size[1]), int(out_feature_size[0])) ### h,w
                    True_False_map = torch.from_numpy(True_False_map)
                    True_False_map = True_False_map.type(FloatTensor)
                    box_info.positive_points_map = True_False_map
        else:
            for box_info in box_info_list:
                if box_info.box_id == box_id:
                    box_info_list.remove(box_info)
    
    bceloss = torch.tensor(0)
    bceloss = bceloss.type(FloatTensor)
    for box_info in box_info_list:
        label_difficulty = box_info.label_difficulty
        label_difficulty = label_difficulty.type(FloatTensor)
        predict_difficulty = difficulty_map * box_info.positive_points_map
        predict_difficulty = torch.max(predict_difficulty)
        bceloss += BCELoss(predict_difficulty, label_difficulty)
    return bceloss


class LossFunc_ThreeBranch(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale=80., stride=2, learn_mode="CPL", soft_label_func="linear", cuda=True, gettargets=False):
        super(LossFunc_ThreeBranch, self).__init__()
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.out_feature_size = [self.model_input_size[0]/stride, self.model_input_size[1]/stride] ## feature_w,feature_h
        self.scale = scale
        self.learn_mode = learn_mode
        self.soft_label_func = soft_label_func
        #(model_input_size, num_classes=2, stride=2)
        self.get_targets = getTargets(model_input_size, num_classes, scale=scale, stride=stride, cuda=True)
        self.cuda = cuda
        self.gettargets = gettargets
    
    def forward(self, input, targets, bboxes):
        # bboxes is a bs list with 'n,c' tensor, n is the num of box. n,6 (6: x1, y1, x2, y2, class_id, object score)
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # targets is bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        if self.gettargets:
            if self.learn_mode == "All_Sample":
                targets = self.get_targets(input, targets, difficult_mode=0) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "SLW" and self.soft_label_func == "linear":
                targets = self.get_targets(input, targets, difficult_mode=1) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            elif self.learn_mode == "SLW" and self.soft_label_func == "piecewise":
                targets = self.get_targets(input, targets, difficult_mode=2) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).
            else:
                raise("Error! learn_mode error.")

        # input is a list with with 3 members(CONF, LOC and DIF), each member is a 'bs,c,in_h,in_w' format tensor).
        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/stride (stride = 2)
        in_w = input[0].size(3) # in_w

        # Branch for task, there are 3 tasks, that is CONF(CONFidence), LOC(LOCation), and DIF(DIFficulty).
        # To get 3D tensor 'bs, in_h, in_w' or 4D tensor 'bs, in_h, in_w, c'.
        #################CONF
        # 
        predict_CONF = input[0].type(FloatTensor) #bs,c,in_h,in_w  c=1
        predict_CONF = predict_CONF.view(bs,in_h,in_w) #bs,in_h,in_w
        ### bs, in_h, in_w
        predict_CONF = torch.sigmoid(predict_CONF)


        #################LOC

        # bs, in_h, in_w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, in_w, in_h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, in_w, in_h -> bs, in_h, in_w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_LOC = input[1].type(FloatTensor) #bs, c,in_h,in_w  c=4(dx1,dy1,dx2,dy2)
        # bs, c, in_h, in_w -> bs, in_h, in_w, c
        predict_LOC = predict_LOC.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale + ref_point_ys

        ### bs, in_h, in_w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_LOC[..., 2] = predict_LOC[..., 2] - predict_LOC[..., 0]
        predict_LOC[..., 3] = predict_LOC[..., 3] - predict_LOC[..., 1]
        predict_LOC[..., 0] = predict_LOC[..., 0] + predict_LOC[..., 2]/2
        predict_LOC[..., 1] = predict_LOC[..., 1] + predict_LOC[..., 3]/2
        ###########################


        predict_Difficulty = input[2].type(FloatTensor) #bs,c,in_h,in_w  c=1
        predict_Difficulty = predict_Difficulty.view(bs,in_h,in_w) #bs,in_h,in_w
        ### bs, in_h, in_w
        predict_Difficulty = torch.sigmoid(predict_Difficulty)

        # targets is a list wiht 2 members, each is a 'bs*in_h,in_w*c' format tensor(cls and bbox).
        # 2,bs*c*in_h,in_w -> 3,bs,in_h,in_w,c (a list with 3 members, each member is a 'bs,in_h,in_w,c' format tensor).

        #################CONF_CLS
        ### bs, in_h, in_w, c(c=num_classes(Include background))
        label_CONF_CLS = targets[0].type(FloatTensor) #bs*in_h,in_w*c  c=num_classes(Include background)
        label_CONF_CLS = label_CONF_CLS.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c
        ### bs, in_h, in_w
        # print("label_CONF_CLS[:,:,:,1:].size()")
        # print(label_CONF_CLS[:,:,:,1:].size())
        label_CONF = torch.sum(label_CONF_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Guassian Heat Conf
        # print("label_CONF.size()")
        # print(label_CONF.size())

        label_CLS_weight =  torch.ceil(label_CONF_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_neg = weight_neg.squeeze(3)
        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)
        ### bs
        bs_neg_nums = torch.sum(weight_neg, dim=(1,2))
        ### bs
        bs_obj_nums = torch.sum(weight_pos, dim=(1,2))
        
        #################LOC
        label_LOC_sampleweight_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_sampleweight_lamda = label_LOC_sampleweight_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_sampleweight_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
        ### bs, in_h, in_w
        label_sampleweight = label_LOC_sampleweight_lamda[:,:,:,4] # bs,in_h,in_w
        ### bs, in_h, in_w
        # label_lamda = label_LOC_sampleweight_lamda[:,:,:,5] # bs,in_h,in_w

        ## Conf Loss
        ## bs, in_h, in_w
        # print("predict_CONF[predict_CONF>0.2]")
        # print(predict_CONF[predict_CONF>0.2])
        MSE_Loss = MSELoss(label_CONF, predict_CONF)
        neg_MSE_Loss = MSE_Loss * weight_neg
        pos_MSE_Loss = (MSE_Loss * label_sampleweight) * weight_pos

        CONF_loss = 0
        for b in range(bs):
            CONF_loss_per_batch = 0
            ### in_h, in_w
            if bs_obj_nums[b] != 0:
                k = bs_obj_nums[b].cpu()
                k = int(k.numpy())
                topk = 2*k
                if topk > bs_neg_nums[b]:
                    topk = bs_neg_nums[b]
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk((neg_MSE_Loss[b]).view(-1), topk).values)
                pos_MSE_Loss_sum = torch.sum(pos_MSE_Loss[b])
                CONF_loss_per_batch = (neg_MSE_Loss_topk_sum + 10*pos_MSE_Loss_sum)/bs_obj_nums[b]
            else:
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk(neg_MSE_Loss[b], 20).values)
                CONF_loss_per_batch = neg_MSE_Loss_topk_sum/10
            CONF_loss += CONF_loss_per_batch
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        ciou_loss = (ciou_loss.view(bs,in_h,in_w)) * label_sampleweight * weight_pos
        LOC_loss = 0
        for b in range(bs):
            LOC_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                LOC_loss_per_batch = torch.sum(ciou_loss[b])/bs_obj_nums[b]
            else:
                LOC_loss_per_batch = 0
            LOC_loss += LOC_loss_per_batch
    
        ### Difficulty Loss (soft label)
        DIF_loss = 0
        for b in range(bs):
            DIF_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                ### the parameters of difficulty_loss_per_batch is: difficulty_map, bboxes, out_feature_size, image_size, FloatTensor
                DIF_loss_per_batch = difficulty_loss_per_batch(predict_Difficulty[b], bboxes[b], self.out_feature_size, self.model_input_size, FloatTensor)/bs_obj_nums[b]
            else:
                DIF_loss_per_batch = 0
            DIF_loss += DIF_loss_per_batch


        total_loss = (10*CONF_loss + 100*LOC_loss + 5*DIF_loss) / bs
        return total_loss