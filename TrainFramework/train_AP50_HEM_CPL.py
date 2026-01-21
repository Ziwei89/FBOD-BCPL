#-------------------------------------#
#       Train the model with loss-based learning strategy, such as HEM、loss-based CPL etc.
#-------------------------------------#
import os
from config.opts import opts
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net.FBODInferenceNet import FBODInferenceBody
from utils.FBODLoss import LossFunc
from FB_detector import FB_Postprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import FBObj
from dataloader.dataset_bbox import CustomDataset, dataset_collate
from utils.common import load_data_raw_resize_boxes, GetMiddleImg_ModelInput_for_MatImageList
from utils.get_box_info_list import getBoxInfoListForOneImage_Loss, image_info
from mAP import mean_average_precision
import copy
import shutil
import random
import math

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def adjust_cpl_threshold(lambda0=0.2, e1=0.1, e2=0.9, step_proportion=0.01, r=1):
    """
    cpl_threshold, cpl based on loss

    """
    if step_proportion <= e1:
        return lambda0
    elif step_proportion <= e2:
        return (1-(1-lambda0)*((e2-step_proportion)/(e2-e1))**r)
    else:
        return 1.

def cpl_sampleWeight_Hard(sample_loss, cpl_threshold):
    if sample_loss < cpl_threshold:
        return 1
    else:
        return 0

def cpl_sampleWeight_Linear(sample_loss, cpl_threshold):
    if sample_loss < cpl_threshold:
        return (1 - sample_loss/cpl_threshold)
    else:
        return 0

def cpl_sampleWeight_Logarithmic(sample_loss, cpl_threshold):
    if sample_loss < cpl_threshold:
        parameter2 = 1- cpl_threshold
        weight = (math.log(sample_loss + parameter2)/math.log(parameter2+0.01))
        return weight
    else:
        return 0

def is_tensor(var):
    return torch.is_tensor(var)
#---------------------------------------------------#
#   Get the classes
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class LablesToResults(object):
    def __init__(self, batch_size):#h,w
        self.batch_size = batch_size

    def covert(self, labels_list, iteration): # TO Raw image size
        label_obj_list = []
        for batch_id in range(self.batch_size):
            labels = labels_list[batch_id]
            if labels.size==0:
                continue
            image_id = self.batch_size*iteration + batch_id
            for label in labels:
                # class_id = label[4] + 1 ###Include background in this project, the label didn't include background classes.
                box = [label[i] for i in range(4)]
                label_obj_list.append(FBObj(score=1., image_id=image_id, bbox=box))
        return label_obj_list

def fit_one_epoch(largest_AP_50,net,model,optimizer,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,gen,genval,
                  Epoch,cuda,save_model_dir,labels_to_results,detect_post_process):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets, names = batch[0], batch[1], batch[2]
            #print(images.shape) 1,7,384,672
            with torch.no_grad():
                # if cuda:
                #     images = Variable(images).to(torch.device('cuda:0'))
                #     targets = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets] ## 
                # else:
                #     images = Variable(images)
                #     targets = [Variable(fature_label)for fature_label in targets] ##
                if cuda:
                    images = Variable(torch.from_numpy(images)).to(torch.device('cuda:0'))
                    targets = [Variable(torch.from_numpy(fature_label)) for fature_label in targets] ## 
                else:
                    images = Variable(torch.from_numpy(images))
                    targets = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets] ##
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_func_train(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'train_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)

            start_time = time.time()
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        all_label_obj_list = []
        all_obj_result_list = []
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]
            labels_list = copy.deepcopy(targets_val)
            with torch.no_grad():
                # if cuda:
                #     images_val = Variable(images_val).to(torch.device('cuda:0'))
                #     targets_val = [Variable(fature_label.to(torch.device('cuda:0'))) for fature_label in targets_val] ## 
                # else:
                #     images_val = Variable(images_val)
                #     targets_val = [Variable(fature_label)for fature_label in targets_val] ##
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val)).to(torch.device('cuda:0'))
                    targets_val = [Variable(torch.from_numpy(fature_label)) for fature_label in targets_val] ## 
                else:
                    images_val = Variable(torch.from_numpy(images_val))
                    targets_val = [Variable(torch.from_numpy(fature_label).type(torch.FloatTensor)) for fature_label in targets_val] ##
                optimizer.zero_grad()
                outputs = net(images_val)
                loss = loss_func_val(outputs, targets_val)
                # print(loss.item())
                val_loss += loss

                if (epoch+1) >= 30:
                    label_obj_list = labels_to_results.covert(labels_list, iteration)
                    all_label_obj_list += label_obj_list

                    obj_result_list = detect_post_process.Process(outputs, iteration)
                    all_obj_result_list += obj_result_list

            pbar.set_postfix(**{'val_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    if (epoch+1) >= 30:
        AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    else:
        AP_50,REC_50,PRE_50 = 0,0,0
    
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1),  AP_50, REC_50, PRE_50))
    
    if (epoch+1)%10 == 0 or epoch == 0:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
        # print('Saving state, iter:', str(epoch+1))
        torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
    else:
        if largest_AP_50 < AP_50:
            largest_AP_50 = AP_50
            # print('Saving state, iter:', str(epoch+1))
            torch.save(model.state_dict(), save_model_dir + 'Epoch%d-Total_Loss%.4f-Val_Loss%.4f-AP_50_%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1),AP_50))
    torch.save(model.state_dict(), save_model_dir + 'FB_object_detect_model.pth')
    if (epoch+1) >= 30:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, AP_50
    else:
        return total_loss/(epoch_size+1), val_loss/(epoch_size_val+1), largest_AP_50, 0.80

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

####################### Plot figure #######################################
def draw_curve_loss(start_log_epoch, epoch, train_loss_list, val_loss_list, pic_name):
    fig = plt.figure()

    ax0 = fig.add_subplot(111, title="Loss curve during training the FB_object_detect model")
    ax0.set_ylabel('loss')
    ax0.set_xlabel('Epochs')

    x_epoch = [i for i in range(start_log_epoch, epoch+1)]

    ax0.plot(x_epoch, train_loss_list, 'b-', label='train')
    ax0.plot(x_epoch, val_loss_list, 'r-', label='val')

    ax0.legend()
    fig.savefig(pic_name, bbox_inches='tight')
    plt.close()
########============================================================########
def draw_curve_ap50(start_log_epoch, epoch, AP50_list, pic_name):
    fig = plt.figure()

    ax0 = fig.add_subplot(111, title="AP$_{50}$ curve during training the FB_object_detect model")
    ax0.set_ylabel('AP$_{50}$')
    ax0.set_xlabel('Epochs')

    x_epoch = [i for i in range(start_log_epoch, epoch+1)]

    ax0.plot(x_epoch, AP50_list, 'g', label='AP$_{50}$')

    ax0.legend()
    fig.savefig(pic_name, bbox_inches='tight')
    plt.close()
#############################################################################

if __name__ == "__main__":

    opt = opts().parse()
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! assign_method error.")
    
    base_Add_name = opt.Add_name

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
        if opt.MF_para == "1-3": # The parameter of the Minimize Function
            MF_para = 1.0/3
        else:
            MF_para = float(opt.MF_para)
        
        if opt.TS_para == "1-3": # The parameter of the Training Scheduling
            TS_para = 1.0/3
        else:
            TS_para = float(opt.TS_para)
    elif opt.learn_mode == "CPL": ### cpl based on loss
        Add_name = opt.cpl_mode + "_" + Add_name
    else:
        Add_name = Add_name
    
    

    
    
    save_model_dir_a = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelA/"
    os.makedirs(save_model_dir_a, exist_ok=True)

    save_model_dir_b = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + Add_name + "_modelB/"
    os.makedirs(save_model_dir_b, exist_ok=True)

    if opt.prior_way == "ASP" or opt.prior_way == "ESP":
        prior_save_model_dir_a = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + prior_learn_mode + "_" + abbr_assign_method \
                                + "_modelA/"
        prior_save_model_dir_b = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                                + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + prior_learn_mode + "_" + abbr_assign_method \
                                + "_modelB/"
        if opt.pretrain_model_name_a == "None":
            pretrain_model_name_a = "FB_object_detect_model.pth"
        else:
            pretrain_model_name_a = opt.pretrain_model_name_a
        if opt.pretrain_model_name_b == "None":
            pretrain_model_name_b = "FB_object_detect_model.pth"
        else:
            pretrain_model_name_b = opt.pretrain_model_name_b
        shutil.copy(prior_save_model_dir_a + pretrain_model_name_a, save_model_dir_a + pretrain_model_name_a)
        shutil.copy(prior_save_model_dir_b + pretrain_model_name_b, save_model_dir_b + pretrain_model_name_b)

    ############### For log figure ################
    log_pic_name_loss_a = save_model_dir_a + "loss.jpg"
    log_pic_name_ap50_a = save_model_dir_a + "ap50.jpg"

    log_pic_name_loss_b = save_model_dir_b + "loss.jpg"
    log_pic_name_ap50_b = save_model_dir_b + "ap50.jpg"
    ################################################

    config_txt = save_model_dir_a + "config.txt"
    if os.path.exists(config_txt):
        pass
    else:
        config_txt_file = open(config_txt, 'w')
        config_txt_file.write("Input mode: " + opt.input_mode + "\n")
        config_txt_file.write("Data root path: " + opt.data_root_path + "\n")
        config_txt_file.write("Aggregation method: " + opt.aggregation_method + "\n")
        config_txt_file.write("Backbone name: " + opt.backbone_name + "\n")
        config_txt_file.write("Fusion method: " + opt.fusion_method + "\n")
        config_txt_file.write("Assign method: " + opt.assign_method + "\n")
        config_txt_file.write("Scale factor: " + str(opt.scale_factor) + "\n")
        config_txt_file.write("Batch size: " + str(opt.Batch_size) + "\n")
        config_txt_file.write("Data augmentation: " + str(opt.data_augmentation) + "\n")
        config_txt_file.write("Load pretrain model: " + str(opt.load_pretrain_model) + "\n")
        config_txt_file.write("Learn rate: " + str(opt.lr) + "\n")
        config_txt_file.write("Learn mode: " + opt.learn_mode + "\n")
        if opt.learn_mode == "CPLBC":
            config_txt_file.write("The parameter of the Minimize Function: " + str(MF_para) + "\n")
            config_txt_file.write("The parameter of the Training Scheduling: " + str(TS_para) + "\n")
        if opt.learn_mode == "CPL": ### cpl based on loss
            config_txt_file.write("CPL mode: " + opt.cpl_mode + "\n")
        config_txt_file.write("Pretrain model a: " + opt.prior_save_model_dir_a + pretrain_model_name_a + "\n")
        config_txt_file.write("Pretrain model b: " + opt.prior_save_model_dir_b + pretrain_model_name_b + "\n")
        config_txt_file.close()

    #-------------------------------#
    #-------------------------------#
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W
    
    Cuda = True
    ##################shuffle train txt to two dataset ###########################
    train_img_label_txt_file_raw = "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    train_annotation_path_a = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + base_Add_name + "_subsetAllA.txt"
    train_annotation_path_b = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + base_Add_name + "_subsetAllB.txt"
    if os.path.exists(train_annotation_path_a):
        pass
    else:
        os.makedirs("./variable_score/", exist_ok=True)
        out = open(train_annotation_path_a, "w")
        lines = []
        with open(train_img_label_txt_file_raw, "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()
    
    if os.path.exists(train_annotation_path_b):
        pass
    else:
        os.makedirs("./variable_score/", exist_ok=True)
        out = open(train_annotation_path_b, "w")
        lines = []
        with open(train_img_label_txt_file_raw, "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()
    ##################shuffle train txt to two dataset ###########################


    train_dataset_image_path = opt.data_root_path + "images/train/"
    
    val_annotation_path =  "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_val.txt"
    val_dataset_image_path = opt.data_root_path + "images/val/"

    #-------------------------------#
    # 
    #-------------------------------#
    classes_path = 'model_data/classes.txt'   
    class_names = get_classes(classes_path)
    num_classes = len(class_names) + 1 #### Include background
    
    # create model
    ### FBODInferenceBody parameters:
    ### input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
    ### backbone_name="cspdarknet53": ### Extract parameters. input_channels equal to aggregation_output_channels.
    model_a = FBODInferenceBody(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels,
                              aggregation_method=opt.aggregation_method, input_mode=opt.input_mode, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)
    
    model_b = FBODInferenceBody(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels,
                              aggregation_method=opt.aggregation_method, input_mode=opt.input_mode, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)

    #-------------------------------------------#
    #   load model
    #-------------------------------------------#
    if opt.start_Epoch !=0 or opt.load_pretrain_model:
        print('Loading weights into state dict...')
        if Cuda:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        ##################### load model A #############################
        pretrain_model_path_a = save_model_dir_a + pretrain_model_name_a
        if not os.path.exists(pretrain_model_path_a):
            raise ValueError(f"Error! No pretrain model: {pretrain_model_path_a}")
        else:
            model_dict = model_a.state_dict()
            pretrained_dict = torch.load(pretrain_model_path_a, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            model_a.load_state_dict(model_dict)
        
        ##################### load model B #############################
        pretrain_model_path_b = save_model_dir_b + pretrain_model_name_b
        if not os.path.exists(pretrain_model_path_b):
            raise ValueError(f"Error! No pretrain model: {pretrain_model_path_b}")
        else:
            model_dict = model_b.state_dict()
            pretrained_dict = torch.load(pretrain_model_path_b, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            model_b.load_state_dict(model_dict)
        print('Finished loading pretrained model!')
    else:
        print('Train the model from scratch!')

    net_a = model_a.train()
    if Cuda:
        net_a = torch.nn.DataParallel(net_a)
        cudnn.benchmark = True
        net_a = net_a.cuda()
    
    net_b = model_b.train()
    if Cuda:
        net_b = torch.nn.DataParallel(net_b)
        cudnn.benchmark = True
        net_b = net_b.cuda()

    # Creat loss function
    # dynamic label assign, so the gettargets is ture.
    loss_func_train = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode=opt.learn_mode, MF_para=MF_para, cuda=Cuda, gettargets=True)
    
    loss_func_val = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode="All_Sample", cuda=Cuda, gettargets=True)


    # For calculating the AP50
    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)

    get_box_info_for_one_image = getBoxInfoListForOneImage_Loss(num_classes=num_classes, image_size = (model_input_size[1],model_input_size[0]),
                                                                scale=opt.scale_factor, cuda=Cuda) # image_size w,h

    with open(train_annotation_path_a) as f:
        train_lines_a = f.readlines()
        num_train_a = len(train_lines_a)
    with open(train_annotation_path_b) as f:
        train_lines_b = f.readlines()
        num_train_b = len(train_lines_b)

    with open(val_annotation_path) as f:
        val_lines = f.readlines()
        num_val = len(val_lines)


    
    #------------------------------------------------------#
    #------------------------------------------------------#
    lr = opt.lr
    Batch_size = opt.Batch_size
    start_Epoch = opt.start_Epoch
    lr = lr*((0.95)**start_Epoch)
    end_Epoch = opt.end_Epoch

    optimizer_a = optim.Adam(net_a.parameters(),lr,weight_decay=5e-4)
    lr_scheduler_a = optim.lr_scheduler.StepLR(optimizer_a,step_size=1,gamma=0.95)

    optimizer_b = optim.Adam(net_b.parameters(),lr,weight_decay=5e-4)
    lr_scheduler_b = optim.lr_scheduler.StepLR(optimizer_b,step_size=1,gamma=0.95)
    
    train_data_a = CustomDataset(train_lines_a, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                               input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
    train_dataloader_a = DataLoader(train_data_a, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)

    train_data_b = CustomDataset(train_lines_b, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                               input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
    train_dataloader_b = DataLoader(train_data_b, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
    

    val_data = CustomDataset(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path, \
                             input_mode=opt.input_mode, continues_num=opt.input_img_num)
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)


    epoch_size = max(1, num_train_a//Batch_size)
    epoch_size_val = num_val//Batch_size
    

    largest_AP_50_a=0
    largest_AP_50_b=0
    train_loss_list_a = []
    val_loss_list_a = []
    ap_50_list_a = []

    train_loss_list_b = []
    val_loss_list_b = []
    ap_50_list_b = []
    for epoch in range(start_Epoch,end_Epoch):
        if opt.learn_mode == "CPL" or opt.learn_mode == "HEM":
            ################ Use model A to update object weight for model B ##########################
            net_a = net_a.eval()
            print("Use model A to update object weight for model B")
            image_info_list = []
            with tqdm(total=len(train_lines_b)) as pbar:
                for line in train_lines_b:
                    images, raw_bboxes, bboxes, first_img_name = load_data_raw_resize_boxes(line, train_dataset_image_path, frame_num=opt.input_img_num, image_size=model_input_size)
                    image_info_instance = image_info(iname=first_img_name)
                    if len(bboxes) == 0:
                        image_info_list.append(image_info_instance)
                        pbar.update(1)
                        continue
                    _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=opt.input_img_num, input_mode=opt.input_mode)
                    with torch.no_grad():
                        model_input = torch.from_numpy(model_input)
                        if Cuda:
                            model_input = model_input.cuda()
                        predictions = net_a(model_input)
                    ### sample loss
                    image_info_instance.box_info_list = get_box_info_for_one_image(predictions, raw_bboxes, bboxes)
                    image_info_list.append(image_info_instance)
                    pbar.update(1)
            ### Update the object weight by rewriting all the information.
            annotation_file = open(train_annotation_path_b,'w')
            if opt.learn_mode == "CPL":
                cpl_threshold = adjust_cpl_threshold((epoch*1.)/end_Epoch)
                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if opt.cpl_mode == "hard":
                                sample_weight = cpl_sampleWeight_Hard(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            elif opt.cpl_mode == "linear":
                                sample_weight = cpl_sampleWeight_Linear(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            elif opt.cpl_mode == "logarithmic":
                                sample_weight = cpl_sampleWeight_Logarithmic(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            else:
                                raise("Error, no such spl mode.")
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            elif opt.learn_mode == "HEM":
                sample_loss_list = []
                for image_info_instance in image_info_list:
                    if len(image_info_instance.box_info_list) == 0:
                        continue
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            sample_loss_list.append(box_info_instance.sample_loss)
                sample_loss_list.sort(reverse=True)
                loss_threshold = sample_loss_list[int(len(sample_loss_list) * 0.4)]

                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if box_info_instance.sample_loss >= loss_threshold:
                                sample_weight = 1
                            else:
                                sample_weight = 0
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            annotation_file.close()
            
            with open(train_annotation_path_b) as f:
                train_lines_b = f.readlines()
            train_data_b = CustomDataset(train_lines_b, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                                    input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
            train_dataloader_b = DataLoader(train_data_b, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
            ###########################################################################################
            ################ Use model B to update object weight for model A ##########################
            net_b = net_b.eval()
            print("Use model B to update object weight for model A")
            image_info_list = []
            with tqdm(total=len(train_lines_a)) as pbar:
                for line in train_lines_a:
                    images, raw_bboxes, bboxes, first_img_name = load_data_raw_resize_boxes(line, train_dataset_image_path, frame_num=opt.input_img_num, image_size=model_input_size)
                    image_info_instance = image_info(iname=first_img_name)
                    if len(bboxes) == 0:
                        image_info_list.append(image_info_instance)
                        pbar.update(1)
                        continue
                    _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=opt.input_img_num, input_mode=opt.input_mode)
                    with torch.no_grad():
                        model_input = torch.from_numpy(model_input)
                        if Cuda:
                            model_input = model_input.cuda()
                        predictions = net_b(model_input)
                    ### sample loss
                    image_info_instance.box_info_list = get_box_info_for_one_image(predictions, raw_bboxes, bboxes)
                    image_info_list.append(image_info_instance)
                    pbar.update(1)
            ### Update the object weight by rewriting all the information.
            annotation_file = open(train_annotation_path_a,'w')
            if opt.learn_mode == "CPL":
                spl_threshold = adjust_cpl_threshold((epoch*1.)/end_Epoch)
                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if opt.cpl_mode == "hard":
                                sample_weight = cpl_sampleWeight_Hard(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            elif opt.cpl_mode == "linear":
                                sample_weight = cpl_sampleWeight_Linear(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            elif opt.cpl_mode == "logarithmic":
                                sample_weight = cpl_sampleWeight_Logarithmic(sample_loss=box_info_instance.sample_loss, cpl_threshold=cpl_threshold)
                            else:
                                raise("Error, no such spl mode.")
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            elif opt.learn_mode == "HEM":
                sample_loss_list = []
                for image_info_instance in image_info_list:
                    if len(image_info_instance.box_info_list) == 0:
                        continue
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            sample_loss_list.append(box_info_instance.sample_loss)
                sample_loss_list.sort(reverse=True)
                loss_threshold = sample_loss_list[int(len(sample_loss_list) * 0.4)]

                for image_info_instance in image_info_list:
                    annotation_file.write(image_info_instance.iname)
                    if len(image_info_instance.box_info_list) == 0:
                        annotation_file.write(" None")
                    else:
                        for box_info_instance in image_info_instance.box_info_list:
                            if box_info_instance.sample_loss >= loss_threshold:
                                sample_weight = 1
                            else:
                                sample_weight = 0
                            string_label = " " + ",".join(str(int(a)) for a in box_info_instance.bbox) + "," + str(int(box_info_instance.class_id)) + "," + str(sample_weight)
                            annotation_file.write(string_label)
                    annotation_file.write("\n")
            annotation_file.close()
            
            with open(train_annotation_path_a) as f:
                train_lines_a = f.readlines()
            train_data_a = CustomDataset(train_lines_a, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                                    input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
            train_dataloader_a = DataLoader(train_data_a, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
            ###########################################################################################
        #################### train model B #####################
        print("Train the model b")
        net_b = net_b.train()
        train_loss_b, val_loss_b,largest_AP_50_record_b, AP_50_b = fit_one_epoch(largest_AP_50_b,net_b,model_b,optimizer_b,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,train_dataloader_b,val_dataloader,
                                                                            end_Epoch,Cuda,save_model_dir_b, labels_to_results=labels_to_results, detect_post_process=detect_post_process)
        largest_AP_50_b = largest_AP_50_record_b
        if (epoch+1)>=2:
            train_loss_list_b.append(train_loss_b.item())
            val_loss_list_b.append(val_loss_b.item())
            draw_curve_loss(start_log_epoch=2, epoch=epoch+1, train_loss_list=train_loss_list_b, val_loss_list=val_loss_list_b, pic_name=log_pic_name_loss_b)
        if (epoch+1)>=30:
            ap_50_list_b.append(AP_50_b.item())
            draw_curve_ap50(start_log_epoch=30, epoch=epoch+1, AP50_list=ap_50_list_b, pic_name=log_pic_name_ap50_b)
        lr_scheduler_b.step()
        #################### train model A #####################
        print("Train the model a")
        net_a = net_a.train()
        train_loss_a, val_loss_a,largest_AP_50_record_a, AP_50_a = fit_one_epoch(largest_AP_50_a,net_a,model_a,optimizer_a,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,train_dataloader_a,val_dataloader,
                                                                            end_Epoch,Cuda,save_model_dir_a, labels_to_results=labels_to_results, detect_post_process=detect_post_process)
        largest_AP_50_a = largest_AP_50_record_a
        if (epoch+1)>=2:
            train_loss_list_a.append(train_loss_a.item())
            val_loss_list_a.append(val_loss_a.item())
            draw_curve_loss(start_log_epoch=2, epoch=epoch+1, train_loss_list=train_loss_list_a, val_loss_list=val_loss_list_a, pic_name=log_pic_name_loss_a)
        if (epoch+1)>=30:
            ap_50_list_a.append(AP_50_a.item())
            draw_curve_ap50(start_log_epoch=30, epoch=epoch+1, AP50_list=ap_50_list_a, pic_name=log_pic_name_ap50_a)

        lr_scheduler_a.step()
