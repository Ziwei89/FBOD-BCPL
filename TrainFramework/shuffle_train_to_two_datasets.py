import random
from config.opts import opts
import os

num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
if __name__ == '__main__':

    opt = opts().parse()
    Add_name = opt.Add_name

    train_img_label_txt_file_raw = "./dataloader/img_label_" + num_to_chinese_c_dic[opt.input_img_num] + "_continuous_difficulty_train.txt"
    in_files = [train_img_label_txt_file_raw, train_img_label_txt_file_raw]
    train_img_label_txt_file_subsetAllA = "./variable_score/img_label_" + num_to_chinese_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetAllA.txt"
    train_img_label_txt_file_subsetAllB = "./variable_score/img_label_" + num_to_chinese_c_dic[opt.input_img_num] + "_continuous_difficulty_train_" + Add_name + "_subsetAllB.txt"
    out_files = [train_img_label_txt_file_subsetAllA, train_img_label_txt_file_subsetAllB]

    os.makedirs("./variable_score/", exist_ok=True)

    for in_file, out_file in zip(in_files, out_files):
        out = open(out_file, "w")
        lines = []
        with open(in_file, "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()