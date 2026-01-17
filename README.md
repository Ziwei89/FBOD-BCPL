[简体中文](README_CN.md)  
# CPL-BC
Due to variations in size or background similarity, flying bird targets captured by surveillance cameras exhibit varying degrees of detectability. To mitigate the negative impact of hard samples on the training of bird detection models ([FBOD](https://github.com/Ziwei89/FBOD)), this paper proposes a Confidence-based Collaborative Pacing Learning strategy (CPL-BC) applied during FBOD training. The strategy involves maintaining two structurally identical but differently initialized models that collaborate to select easily detectable samples for training. When prediction confidence exceeds a predefined threshold, the threshold is gradually lowered during training to progressively enhance the model's ability to identify targets from easy to difficult cases. Prior to applying CPL-BC, both FBOD models undergo pretraining to develop the capability to assess sample difficulty. 

This project builds upon the [FBOD](https://github.com/Ziwei89/FBOD) and [FBOD-BSPL](https://github.com/Ziwei89/FBOD-BSPL) projects to propose a novel model training strategy. Most parameter meanings and configurations remain consistent with those in [FBOD](https://github.com/Ziwei89/FBOD) and [FBOD-BSPL](https://github.com/Ziwei89/FBOD-BSPL); thus, they will not be reiterated here. During training, parameters unrelated to the training strategy (default parameters) adopt default settings. For details on these parameters, please refer to: https://github.com/Ziwei89/FBOD 和 https://github.com/Ziwei89/FBOD-BSPL 

Additionally, this project defaults to employing frame padding.  

# Project Application Steps

## 1. Clone the project locally
```
git clone https://github.com/Ziwei89/FBOD-BCPL.git
```
## 2. Prepare training and test data
**<font color=red>Note：</font> Default to project root (FBOD-BCPL/) before running scripts.**

You can use labelImg to annotate the image and get an xml file.  
The label file should contain the object bounding box, the object category, and the difficulty information of the object (we will use the difficulty information of the object in a future project, so don't worry about it in this project, and the relevant code will set a default value that is not needed).

### (1) data organization
```  
data_root_path/  
               videos/
                     train/
                           bird_1.mp4
                           bird_2.mp4
                           ...  
                     val/
                images/
                     train/  
                           bird_1_000000.jpg  
                           bird_1_000001.jpg  
                           ...  
                           bird_2_000000.jpg  
                           ...  
                     val/
                labels
                     train/  
                           bird_1_000000.xml  
                           bird_1_000001.xml  
                           ...
                           bird_2_000000.xml
                           ...  
                     val/
```  
### (2) Generate data description txt file for training and testing (train one txt, test one txt)
The format of the data description txt file is as follows:  
The first frame of n consecutive frames image name *space* middle frame bird target information 
image_name x1,y1,x2,y2,cls,difficulty x1,y1,x2,y2,cls,difficulty  
eg:  
```
...  
bird_3_000143.jpg 995,393,1016,454,0,0.625
bird_3_000144.jpg 481,372,489,389,0,0.375 993,390,1013,456,0,0.625
...  
bird_40_000097.jpg None
...
```
We provide a script that generates such a data description txt. This script adds the sequence padding to continuous_image_annotation_frames_padding.py in the Data_process directory. Sequence padding is to add some black images before and after the beginning and end of the video, so that the first and second frames have output results (please refer to [FBOD paper](https://ieeexplore.ieee.org/document/10614237/) for details). To run this script, we need to specify the dataset path and the number of consecutive image frames to input to the model for one inference: 
```
cd Data_process # Go to the data processing directory from the project root
python continuous_image_annotation.py \
       --data_root_path=../dataset/FBD-SV-2024/ \
       --input_img_num=5
```
After running this script, two txt files will be generated in the TrainFramework/dataloader/ directory, namely img_label_five_continuous_difficulty_train_raw.txt and img_label_five_continuous_difficulty_val_raw.txt. The training samples in these two files are arranged in sequence. It is recommended to run the following script to shuffle them. 
```
cd TrainFramework/dataloader/ # From the project root directory into the training framework under the dataloader directory
python shuffle_txt_lines.py \
       --input_img_num=5
```
After running this script, two files, img_label_five_continuous_difficulty_train.txt and img_label_five_continuous_difficulty_val.txt, will be generated in the TrainFramework/dataloader/ directory.
### (3) Prepare the classes txt file
Create a folder named "model_data" under the "TrainFramework/" directory, and then create a file named "classes.txt" in the "TrainFramework/model_data/" directory. This file records the categories, such as: 
```
bird
```

## 3. Different training strategies train the model
Five model training strategies are compared, which are All_Sample training strategy, Easy_Sample training strategy, CPL-BC learning strategy, loss-based CPL learning strategy and difficult sample mining training strategy.  
The script TrainFramework/train_AP50.py is used to train the three model training strategies: All_Sample training strategy, Easy_Sample training strategy and CPL-BC learning strategy. The loss-based CPL-BC learning strategy and difficult sample mining training strategy use the script TrainFramework/train_AP50_HEM_CPL.py 

**<font color=red>NOTE：</font>The CPL-BC learning strategy consists of two training phases. Firstly, the model is trained by using easy samples or using all samples, and then the model is trained by using the CPL-BC learning strategy (the model trained by the Easy_Sample training strategy or the All_Sample training strategy is used as the pre-training model).**

The following is the explanation of the relevant parameters (please refer to the TrainFramework/config/opts.py file when setting): 
```
data_root_path                     # Dataset root path
pretrain_model_path                # The path of the pre-trained model. It is necessary to use the  Easy_Sample training strategy or the All_Sample training strategy to train the model
Add_name                           # Add a suffix to the relevant documentation (such as the model save folder or the training documentation image)
learn_mode                         # Model learning strategy:
                                            All_Sample：General Training Strategy with all samples
                                            Easy_Sample：Training Strategy with easy samples
                                            CPLBC：Confidence-based Collaborative Pacing Learning strategy
                                            CPL：Loss-based Collaborative Pacing Learning strategy
                                            HEM：Model Training strategy for mining difficult samples
cpl_mode                            # Self-paced regularizer, effective when loss-based cooperative paced learning strategy: hard, linear, logarithmic
prior_way                           # Prior way: ASP or ESP, that is, All Sample Prior or Easy Sample Prior
```
The other two parameters, MF_para and TS_para, are about the minimization function and the training scheduling function. To be consistent with the presentation of the paper, keep using the default parameters. 

Three examples of training: 
* General Training Strategy with all samples (All_Sample)： 
```
cd TrainFramework
python3 train_AP50.py \
        --data_augmentation \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --start_Epoch=0 \
        --end_Epoch=50 \
        --learn_mode=All_Sample \
        --Add_name=20241127
cd ../
```

* Confidence-based Collaborative Pacing Learning strategy (CPL-BC) with ASP： 
Before the model is trained, a variable_score folder is automatically created, which contains two label files (these are mutually exclusive and used by both models for training). As both models train, they modify each other's sample scores.
```
cd TrainFramework
python train_AP50.py \
        --data_augmentation \
        --pretrain_model_name_a=FB_object_detect_model_a.pth \
        --pretrain_model_name_b=FB_object_detect_model_b.pth \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --start_Epoch=0 \
        --end_Epoch=100 \
        --prior_way=ASP \
        --learn_mode=CPLBC \
        --Add_name=20241220
cd ../
```
* Loss-based Collaborative Pacing Learning strategy 
```
cd TrainFramework
python train_AP50_HEM_SPL.py \
        --data_augmentation \
        --pretrain_model_name_a=FB_object_detect_model_a.pth \
        --pretrain_model_name_b=FB_object_detect_model_b.pth \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --start_Epoch=0 \
        --end_Epoch=100 \
        --prior_way=ASP \
        --learn_mode=CPL \
        --cpl_mode=hard \
        --Add_name=20241220
cd ../
```
## 4. Test model detection performance (test the model with the same parameters as the model was trained with)
```
cd TrainFramework
python mAP_for_AllVideo_coco_tools.py \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --prior_way=ASP \
        --learn_mode=CPLBC \
        --Add_name=20240104 \
        --modelAorB=modelB \
        --model_name=FB_object_detect_model.pth
cd ../
```
