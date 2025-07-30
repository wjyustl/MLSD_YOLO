import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def liang_barsky_clip(x_min, y_min, x_max, y_max, x1, y1, x2, y2):
    """使用Liang-Barsky算法裁剪线段到矩形区域"""
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
    u1, u2 = 0.0, 1.0

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return None  # 线段平行且在外侧
        else:
            r = q[i] / p[i]
            if p[i] < 0:
                if r > u2:
                    return None
                if r > u1:
                    u1 = r
            else:
                if r < u1:
                    return None
                if r < u2:
                    u2 = r

    if u1 > u2:
        return None

    x1_clip = x1 + u1 * dx
    y1_clip = y1 + u1 * dy
    x2_clip = x1 + u2 * dx
    y2_clip = y1 + u2 * dy

    return x1_clip, y1_clip, x2_clip, y2_clip


def process_tile(image, json_data, tile_x, tile_y, crop_size):
    """处理单个切割区域"""
    # 计算当前切片的边界
    crop_width, crop_height = crop_size
    x_min = tile_x * crop_width
    y_min = tile_y * crop_height
    x_max = min((tile_x + 1) * crop_width, image.shape[1])
    y_max = min((tile_y + 1) * crop_height, image.shape[0])

    # 提取图像切片
    tile_img = image[y_min:y_max, x_min:x_max]

    # 准备新JSON数据
    new_json = {
        "version": json_data["version"],
        "flags": json_data["flags"],
        "shapes": [],
        "imagePath": "",
        "imageData": None,
        "imageHeight": tile_img.shape[0],
        "imageWidth": tile_img.shape[1]
    }

    # 处理每个标注
    for shape in json_data["shapes"]:
        if shape["shape_type"] != "line":
            continue  # 跳过非直线标注

        x1, y1 = shape["points"][0]
        x2, y2 = shape["points"][1]

        # 裁剪线段到当前切片
        clipped = liang_barsky_clip(x_min, y_min, x_max, y_max, x1, y1, x2, y2)
        if clipped:
            x1c, y1c, x2c, y2c = clipped
            # 转换为切片局部坐标
            new_shape = shape.copy()
            new_shape["points"] = [
                [x1c - x_min, y1c - y_min],
                [x2c - x_min, y2c - y_min]
            ]
            new_json["shapes"].append(new_shape)

    return tile_img, new_json


def split_image_and_annotations(image_path, json_path, output_dir, crop_size):
    """主处理函数"""
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载图像和JSON
    image = cv2.imread(image_path)
    with open(json_path, encoding='utf-8') as f:
        json_data = json.load(f)

    # 计算切片数量
    height, width = image.shape[:2]
    crop_width, crop_height = crop_size
    # x_tiles = (width + crop_width - 1) // crop_width
    # y_tiles = (height + crop_height - 1) // crop_height
    x_tiles = width // crop_width
    y_tiles = height // crop_height

    # 处理每个切片
    for y in range(y_tiles):
        for x in range(x_tiles):
            tile_img, tile_json = process_tile(image, json_data, x, y, crop_size)

            # 生成文件名
            base_name = f"{Path(image_path).stem}_{y}_{x}"
            img_filename = f"{base_name}.jpg"
            json_filename = f"{base_name}.json"

            # 保存切片图像
            cv2.imwrite(str(Path(output_dir) / img_filename), tile_img)

            # 更新并保存JSON
            tile_json["imagePath"] = img_filename
            with open(Path(output_dir) / json_filename, 'w') as f:
                json.dump(tile_json, f, indent=2)


def crop_image(input_dir, output_dir, crop_size=(640, 640)):
    """
    将大图裁剪为多个不重叠的小图

    参数:
    input_dir: 输入图像路径
    output_dir: 输出文件夹路径
    crop_size: 裁剪尺寸 (width, height)
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.endswith(("jpeg", ".jpg", ".JPG")):
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path)
            width, height = img.size
            crop_width, crop_height = crop_size

            # 计算水平和垂直方向的裁剪数量
            num_horizontal = width // crop_width
            num_vertical = height // crop_height

            # 循环裁剪图像
            for i in range(num_vertical):
                for j in range(num_horizontal):
                    # 计算裁剪区域坐标 (left, upper, right, lower)
                    left = j * crop_width
                    upper = i * crop_height
                    right = left + crop_width
                    lower = upper + crop_height

                    # 裁剪图像
                    cropped_img = img.crop((left, upper, right, lower))

                    # 保存裁剪后的图像
                    output_path = os.path.join(output_dir, f"{img_name.replace('.jpeg', '')}_{i}_{j}.jpeg")
                    cropped_img.save(output_path)


if __name__ == "__main__":
    """切割大图的目的: 训练模型"""
    # 配置参数
    img_path = r"D:\_DATA\taihe_0625\hangdian\DJI_20250625170357_0008.jpg"
    json_path = r"D:\_DATA\taihe_0625\hangdian\DJI_20250625170357_0008.json"
    input_dir = r"D:\_DATA\Emergence Detection\老田15m飞 2025-07-21 17_07_51 (UTC+08)"
    output_dir = r"D:\_DATA\Emergence Detection\老田15m飞 2025-07-21 17_07_51 (UTC+08)\res"
    crop_size = (256, 256)  # 小图尺寸

    # split_image_and_annotations(img_path, json_path, output_dir, crop_size)
    crop_image(input_dir, output_dir, crop_size=crop_size)
    print("\n裁剪完成! 输出位置:", os.path.abspath(output_dir))
