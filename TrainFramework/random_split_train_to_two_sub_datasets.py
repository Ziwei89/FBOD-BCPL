
from config.opts import opts
import os
import random

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

if __name__ == "__main__":
    opt = opts().parse()

    Add_name = opt.Add_name
    
    train_annotation_path = "./dataloader/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    trainA_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetA.txt"
    trainB_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetB.txt"

    os.makedirs("./variable_score/", exist_ok=True)
    bird_ids = []
    train_videos_path = opt.data_root_path + "videos/train/"
    video_names = os.listdir(train_videos_path)
    videos_num = len(video_names)

    for video_name in video_names:
        bird_ids.append(video_name.split(".")[0].split("_")[1])


    trainA_bird_ids = random.sample(bird_ids, int(videos_num/2))
    # print(trainA_bird_ids)
    # print(len(trainA_bird_ids))

    trainA_annotation_file = open(trainA_annotation_path, "w")
    trainB_annotation_file = open(trainB_annotation_path, "w")

    with open(train_annotation_path, "r") as lines:
        for line in lines:
            bird_id = line.split("_")[1]
            # print(bird_id)
            if bird_id in trainA_bird_ids:
                trainA_annotation_file.write(line)
            else:
                trainB_annotation_file.write(line)
    print("Finished!")
    trainA_annotation_file.close()
    trainB_annotation_file.close()