"""
labelme数据转换成
mlsd输入数据格式


img_files_path： 图片和json共同存放目录
writePath： 生成json存储位置
"""

import cv2
from glob import glob

import json

img_files_path = r'D:\_DATA\taihe_0625\hangdian\val'
writePath = r'D:\_DATA\taihe_0625\hangdian\val.json'

img_files = glob('%s/*.json' % img_files_path)

with open(writePath, 'w') as fo:
    label = []
    for i in range(len(img_files)):
        print(
            f"==========================={img_files[i].split('/')[-1][:-5] + '.jpg'}==============================")
        dic = {"filename": img_files[i].split('\\')[-1][:-5] + '.jpg'}

        print(img_files[i].split('.')[0] + '.jpg')
        im = cv2.imread(img_files[i].split('.')[0] + '.jpg')
        h, w, _ = im.shape
        lines = []
        with open(img_files[i], 'r') as fi:
            json_info = json.load(fi)
            for line in json_info["shapes"]:
                # lines.append(line["points"])
                lines.append([item for sublist in line["points"] for item in sublist])
        dic["lines"] = lines
        dic["height"] = h
        dic["width"] = w
        print(dic)
        label.append(dic)
    json.dump(label, fo)
