
from config.opts import opts
import os
import random

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

if __name__ == "__main__":
    opt = opts().parse()

    Add_name = opt.Add_name

    trainA_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetA.txt"
    trainB_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetB.txt"

    trainAll_annotation_path = "./variable_score/img_label_" + num_to_english_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetAll.txt"

    if os.path.exists(trainA_annotation_path) and os.path.exists(trainB_annotation_path):
        pass
    else:
        raise("Error! No train_annotation_path.")
    
    out = open(trainAll_annotation_path, "w")
    lines = []
    with open(trainA_annotation_path, "r") as infile:
        for line in infile:
            lines.append(line)
    with open(trainB_annotation_path, "r") as infile:
        for line in infile:
            lines.append(line)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    for line in lines:
        out.write(line)
    out.close()