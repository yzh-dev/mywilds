# 按照如下方式划分数据集
# RootDir
# └───Domain1Name
# │   └───Class1Name
# │       │   file1.jpg
# │       │   file2.jpg
# │       │   ...
# │   ...
# └───Domain2Name
# |   ...

# %%
import os
import shutil

import pandas as pd
from pathlib import Path


# %%

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
    else:
        print("---  There is this folder!  ---")


# dirname = "D:\ML\Dataset\iwildcamDataset\myiwildcam_v2.0"
# mkdir(dirname+r"\train")  # 调用函数
# %%

def create_dir_split(root_dir, split_dirs):
    # 创建"train", "val", "test","id_val","id_test"文件夹

    df_DataAll = pd.read_csv(root_dir / 'metadata.csv',index_col=0)
    df_categories = pd.read_csv(root_dir / 'categories.csv',index_col=0)

    for split in split_dirs:
        # 创建对应文件夹
        split_path = root_dir / split
        mkdir(split_path)  # 调用函数
        # 近获取对应split的数据
        df_split = df_DataAll[df_DataAll['split'] == split]
        df_split = df_split.reset_index(drop=True)

        # 保存对应split的数据
        # filename = split+'_metadata.csv'
        # df_split.to_csv(split_path / filename, index=False)

        # 创建类别子文件夹
        # categories = df_split['category_id'].unique()
        # for category in categories:
        #     category_name = df_categories[df_categories["category_id"]==category]["name"].values[0]
        #     category_path = split_path / category_name
        #     mkdir(category_path)

        # 将图片移动到对应的类别子文件夹
        origin_wild_path = r"D:\ML\Dataset\iwildcamDataset\iwildcam_v2.0\train"
        origin_wild_path = Path(os.path.abspath(origin_wild_path))
        for index, row in df_split.iterrows():#遍历每一行
            src = origin_wild_path / row['filename']  # 原始路径
            category_name = df_categories[df_categories["category_id"] == row["category_id"]]["name"].values[0]
            dst = split_path / category_name/row['filename']  # 目标路径
            shutil.copy(src, dst)  # 复制文件
        #     # print("复制文件%s到%s" % (src, dst))






if __name__ == '__main__':
    dir = "D:\ML\Dataset\iwildcamDataset\myiwildcam_v2.0\images"
    dirpath = Path(os.path.abspath(dir))
    split_dirs = ["train", "val", "test", "id_val", "id_test"]
    create_dir_split(root_dir=dirpath, split_dirs=split_dirs)

#%%
# 创建长度为323的list，从0开始，步长为1
import numpy as np
locations = list()
for index in range(323):
    locations.append("location" + str(index))
print(locations)



