import os
import random
import shutil
from pathlib import Path


def copy_files(file_list, output_dir):
    for file_name in file_list:
        if file_name.endswith(".jpg"):
            img_path = os.path.join(data_dir, file_name)
            json_name = Path(file_name).with_suffix(".json")
            json_path = os.path.join(data_dir, json_name)

            if not Path(json_path).exists():
                raise FileNotFoundError(f"警告: 找不到 {json_path} 对应的标注文件")

            shutil.copy(img_path, output_dir / Path(file_name))
            shutil.copy(json_path, output_dir / json_name)


def split_labelme_to_TrainVal(data_dir, train_ratio=0.8):
    """
    labelme标注数据集分割
    """
    train_dir = os.path.join(os.path.dirname(data_dir), "train")
    val_dir = os.path.join(os.path.dirname(data_dir), "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    file_list = os.listdir(data_dir)
    random.shuffle(file_list)

    train_list = file_list[:int(len(file_list) * train_ratio)]
    val_list = file_list[int(len(file_list) * train_ratio):]

    copy_files(train_list, train_dir)
    copy_files(val_list, val_dir)


if __name__ == '__main__':
    data_dir = r"D:\_DATA\taihe_0625\hangdian\mlsd_all"
    split_labelme_to_TrainVal(data_dir)
