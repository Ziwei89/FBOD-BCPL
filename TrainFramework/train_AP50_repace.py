#-------------------------------------#
#       对数据集进行训练
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
from utils.get_box_info_list import getBoxInfoListForOneImage
from mAP import mean_average_precision
import copy

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def adjust_spl_threshold(step_proportion=0.01, TS_para=1):
    """
    spl_threshold
    """
    if step_proportion <= 0.1:
        return 0.8
    elif step_proportion <= 0.9:
        return 0.9-step_proportion
    else:
        return 0.

def is_tensor(var):
    return torch.is_tensor(var)
#---------------------------------------------------#
#   获得类
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

def fit_one_epoch(largest_AP_50,net,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,gen,genval,
                  Epoch,cuda,save_model_dir,labels_to_results,detect_post_process,spl_threshold):
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
            loss = loss_func_train(outputs, targets, spl_threshold)
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
    # print('Start Validation')
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
                loss = loss_func_val(outputs, targets_val, spl_threshold)
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
    
    # print('Finish Validation')
    # print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    # print('Total Loss: %.4f || Val Loss: %.4f  || AP_50: %.4f  || REC_50: %.4f  || PRE_50: %.4f' % (total_loss/(epoch_size+1), val_loss/(epoch_size_val+1),  AP_50, REC_50, PRE_50))
    
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
def draw_curve_loss(start_log_epoch, epoch, train_loss_str, val_loss_str, pic_name):
    fig = plt.figure()

    ax0 = fig.add_subplot(111, title="Loss curve during training the FB_object_detect model")
    ax0.set_ylabel('loss')
    ax0.set_xlabel('Epochs')

    x_epoch = [i for i in range(start_log_epoch, epoch+1)]
    train_loss_list = [float(i) for i in train_loss_str.split(",")]
    val_loss_list = [float(i) for i in val_loss_str.split(",")]

    ax0.plot(x_epoch, train_loss_list, 'b-', label='train')
    ax0.plot(x_epoch, val_loss_list, 'r-', label='val')

    ax0.legend()
    fig.savefig(pic_name, bbox_inches='tight')
    plt.close()
########============================================================########
def draw_curve_ap50(start_log_epoch, epoch, ap_50_str, pic_name):
    fig = plt.figure()

    ax0 = fig.add_subplot(111, title="AP$_{50}$ curve during training the FB_object_detect model")
    ax0.set_ylabel('AP$_{50}$')
    ax0.set_xlabel('Epochs')

    x_epoch = [i for i in range(start_log_epoch, epoch+1)]
    AP50_list = [float(i) for i in ap_50_str.split(",")]

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
    
    
    data_subset = opt.data_subset ### subsetA, subsetB or subsetAll.
    modelAorB = opt.modelAorB  #### modelA or modelB
    
    save_model_dir = "logs/" + num_to_english_c_dic[opt.input_img_num] + "/" + opt.model_input_size + "/" + opt.input_mode + "_" + opt.aggregation_method \
                             + "_" + opt.backbone_name + "_" + opt.fusion_method + "_" + opt.learn_mode + "_" + abbr_assign_method \
                             + "_"  + opt.MF_para + "_"  + opt.TS_para + "_" + opt.Add_name + "_"  + modelAorB + "/"
    os.makedirs(save_model_dir, exist_ok=True)

    ############### For log figure ################
    log_pic_name_loss = save_model_dir + "loss.jpg"
    log_pic_name_ap50 = save_model_dir + "ap50.jpg"
    ################################################

    if opt.MF_para == "1-3":
        MF_para = 1.0/3
    else:
        MF_para = float(opt.MF_para)
    
    if opt.TS_para == "1-3":
        TS_para = 1.0/3
    else:
        TS_para = float(opt.TS_para)

    config_txt = save_model_dir + "config.txt"
    if os.path.exists(config_txt):
        pass
    else:
        config_txt_file = open(config_txt, 'w')
        config_txt_file.write("Input mode: " + opt.input_mode + "\n")
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
        config_txt_file.write("The parameter of the Minimize Function: " + str(MF_para) + "\n")
        config_txt_file.write("The parameter of the Training Scheduling: " + str(TS_para) + "\n")
        config_txt_file.close()

    #-------------------------------#
    #-------------------------------#
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W
    
    Cuda = True

    train_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + opt.Add_name + "_" + data_subset + ".txt"
    if os.path.exists(train_annotation_path):
        pass
    else:
        raise ValueError(f"Error! No train_annotation_path: {train_annotation_path}")

    # if opt.learn_mode == "SLW":
    #     train_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + base_Add_name + "_" + data_subset + ".txt"
    #     if os.path.exists(train_annotation_path):
    #         pass
    #     else:
    #         raise("Error! No train_annotation_path.")
    # else:
    #     train_annotation_path = "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"

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
    model = FBODInferenceBody(input_img_num=opt.input_img_num, aggregation_output_channels=opt.aggregation_output_channels,
                              aggregation_method=opt.aggregation_method, input_mode=opt.input_mode, backbone_name=opt.backbone_name, fusion_method=opt.fusion_method)

    #-------------------------------------------#
    #   load model
    #-------------------------------------------#
    if opt.start_Epoch !=0 or opt.load_pretrain_model:
        # pretrain_model_path = glob.glob(save_model_dir + "Epoch"+str(opt.start_Epoch)+"-*")[0] ### Has wildcard in model name eg. Epoch20_*
        pretrain_model_path = save_model_dir + "FB_object_detect_model.pth"
        if not os.path.exists(pretrain_model_path):
            raise ValueError(f"Error! No pretrain model: {pretrain_model_path}")
        else:
            if Cuda:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pretrain_model_path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        # Train the model from scratch!
        pass

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    # dynamic label assign, so the gettargets is ture.
    loss_func_train = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode=opt.learn_mode, soft_label_func=opt.soft_label_func, spl_mode=opt.spl_mode, cuda=Cuda, gettargets=True)
    
    loss_func_val = LossFunc(num_classes=num_classes, model_input_size=(model_input_size[1], model_input_size[0]), \
                         learn_mode="Normal", cuda=Cuda, gettargets=True)


    # For calculating the AP50
    detect_post_process = FB_Postprocess(batch_size=opt.Batch_size, model_input_size=model_input_size)
    labels_to_results = LablesToResults(batch_size=opt.Batch_size)

    get_box_info_for_one_image = getBoxInfoListForOneImage(image_size = (model_input_size[1],model_input_size[0])) # image_size w,h

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
        num_train = len(train_lines)
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
    total_Epoch = opt.total_Epoch

    optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    
    train_data = CustomDataset(train_lines, (model_input_size[1], model_input_size[0]), image_path=train_dataset_image_path, \
                               input_mode=opt.input_mode, continues_num=opt.input_img_num, data_augmentation=opt.data_augmentation)
    train_dataloader = DataLoader(train_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)
    
    val_data = CustomDataset(val_lines, (model_input_size[1], model_input_size[0]), image_path=val_dataset_image_path, \
                             input_mode=opt.input_mode, continues_num=opt.input_img_num)
    val_dataloader = DataLoader(val_data, batch_size=Batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset_collate)


    epoch_size = max(1, num_train//Batch_size)
    epoch_size_val = num_val//Batch_size
    

    ## input_train_val_loss_ap50_str: "x1,x2,x3,x4.../y1,y2,y3,y4.../z1,z2,z3,z4.../q1". 
    # x means train_loss, y means val loss, z means ap_50, q means the largest ap_50.
    # If z is None, it means that there is no ap_50.
    train_loss_str=opt.input_train_val_loss_ap50_str.split("/")[0]
    val_loss_str=opt.input_train_val_loss_ap50_str.split("/")[1]
    ap_50_str=opt.input_train_val_loss_ap50_str.split("/")[2]
    largest_AP_50=float(opt.input_train_val_loss_ap50_str.split("/")[3])
    for epoch in range(start_Epoch,end_Epoch):
        if opt.learn_mode == "CPLBC":
            spl_threshold = adjust_spl_threshold((epoch*1.)/total_Epoch, TS_para=TS_para)
        else:
            spl_threshold=None
        train_loss, val_loss,largest_AP_50_record, AP_50 = fit_one_epoch(largest_AP_50,net,loss_func_train,loss_func_val,epoch,epoch_size,epoch_size_val,train_dataloader,val_dataloader,
                                                                            total_Epoch,Cuda,save_model_dir, labels_to_results=labels_to_results, detect_post_process=detect_post_process, spl_threshold=spl_threshold)
        largest_AP_50 = largest_AP_50_record
        if (epoch+1)>=2:
            if train_loss_str == "None":
                train_loss_str = str(train_loss.item())
                val_loss_str = str(val_loss.item())
            else:
                train_loss_str = train_loss_str + "," + str(train_loss.item())
                val_loss_str = val_loss_str + "," + str(val_loss.item())
        if (epoch+1)>=30:
            if ap_50_str == "None":
                ap_50_str = str(AP_50.item())
            else:
                ap_50_str = ap_50_str + "," + str(AP_50.item())
        if (epoch+1)>=2:
            draw_curve_loss(start_log_epoch=2, epoch=epoch+1, train_loss_str=train_loss_str, val_loss_str=val_loss_str, pic_name=log_pic_name_loss)
        if (epoch+1)>=30:
            draw_curve_ap50(start_log_epoch=30, epoch=epoch+1, ap_50_str=ap_50_str, pic_name=log_pic_name_ap50)
        lr_scheduler.step()
    # return_str: "x1,x2,x3,x4.../y1,y2,y3,y4.../z1,z2,z3,z4.../q1". 
    # x means train_loss, y means val loss, z means ap_50, q means the largest ap_50.
    if is_tensor(largest_AP_50):
        return_str = train_loss_str + "/" + val_loss_str + "/" + ap_50_str + "/" + str(largest_AP_50.item())
    else:
        return_str = train_loss_str + "/" + val_loss_str + "/" + ap_50_str + "/" + str(largest_AP_50)
    print(return_str) ### The content of the last print is the return value print to the bash.