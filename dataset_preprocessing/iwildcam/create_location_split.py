# 按照如下方式划分数据集
# RootDir
# └───Location0
# │   └───Class1Name
# │       │   file1.jpg
# │       │   file2.jpg
# │       │   ...
# │   ...
# └───Location1
# |   ...

# %%
import os
import shutil

import numpy as np
import pandas as pd
from pathlib import Path


# %%

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        # print("---  new folder...  ---")
    else:
        # print("---  There is this folder!  ---")
        pass


# dirname = "D:\ML\Dataset\iwildcamDataset\myiwildcam_v2.0"
# mkdir(dirname+r"\train")  # 调用函数
# %%

def create_location_split(root_dir):
    df_DataAll = pd.read_csv(root_dir / 'metadata.csv', index_col=0)
    df_categories = pd.read_csv(root_dir / 'categories.csv')

    # 图片所在文件夹
    images_dir = root_dir / "ImagesByLocations"
    # 总共323个地址，每个地址对应一个domain的文件夹
    locations = list()
    for index in range(323):
        locations.append("location" + str(index))

    # 创建对应文件夹
    for location in locations:
        location_path = images_dir / location
        mkdir(location_path)  # 调用函数

        # 每个location下都创建所有的类别子文件夹
        # for index in range(182):  # 182个类别
        #     name = df_categories[df_categories["y"] == index]["name"].values[0]
        #     category_path = location_path / name
        #     mkdir(category_path)
    print("文件夹创建完成")

    # 将图片复制到对应的location下
    origin_wild_path = r"D:\ML\Dataset\iwildcamDataset\iwildcam_v2.0\train"
    origin_wild_path = Path(os.path.abspath(origin_wild_path))

    for index, row in df_DataAll.iterrows():  # 遍历每一行
        src = origin_wild_path / row['filename']  # 原始路径
        location = "location" + str(row["location_remapped"])  # 图片所在的location
        # category = df_categories[df_categories["category_id"] == row["category_id"]]["name"].values[0]  # 图片所在的类别
        # dst = images_dir / location / category / row['filename']  # 目标路径
        dst = images_dir / location / row['filename']  # 目标路径
        shutil.copy(src, dst)  # 复制文件

    print("文件复制完成")


if __name__ == '__main__':
    dir = "D:\ML\Dataset\iwildcamDataset\myiwildcam_v2.0"
    dirpath = Path(os.path.abspath(dir))
    create_location_split(root_dir=dirpath)
