import json
import os

# category_dict = {'weed': 0}  # 类别字典
category_dict = {'weed': 0}


def json_to_yolo(input_file_path, output_directory):
    data = json.load(open(input_file_path, encoding="utf-8"))  # 读取带有中文的文件
    image_width = data["imageWidth"]  # 获取json文件里图片的宽度
    image_height = data["imageHeight"]  # 获取json文件里图片的高度
    yolo_format_content = ''

    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            # 归一化坐标点，并计算中心点(cx, cy)、宽度和高度
            [[x1, y1], [x2, y2]] = shape['points']
            x1, x2 = x1 / image_width, x2 / image_width
            y1, y2 = y1 / image_height, y2 / image_height
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # 将数据组装成YOLO格式
            line = "%s %.4f %.4f %.4f %.4f\n" % (category_dict[shape['label']], cx, cy, width, height)  # 生成txt文件里每行的内容
            yolo_format_content += line

        if shape['shape_type'] == "point":
            # 获取点坐标 (单个点)
            point = shape['points'][0]
            x, y = point

            # 归一化坐标 (YOLO格式要求)
            cx = x / image_width
            cy = y / image_height

            # 为点创建一个小边界框 (YOLO需要边界框)
            # 使用1像素宽高的框，然后归一化
            width = 1 / image_width
            height = 1 / image_height

            # 将数据组装成YOLO格式
            line = f"{category_dict[shape['label']]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n"
            yolo_format_content += line

    # 生成txt文件的相应文件路径
    output_file_path = os.path.join(output_directory, os.path.basename(input_file_path).replace('json', 'txt'))
    with open(output_file_path, 'w', encoding='utf-8') as file_handle:
        file_handle.write(yolo_format_content)


input_directory = r"D:\_DATA\Weed_Detection\changshu_weed_0729\data_weed"
output_directory = r"D:\_DATA\Weed_Detection\changshu_weed_0729\label"
os.makedirs(output_directory, exist_ok=True)

file_list = os.listdir(input_directory)
json_file_list = [file for file in file_list if file.endswith(".json")]  # 获取所有json文件的路径

for json_file in json_file_list:
    json_to_yolo(os.path.join(input_directory, json_file), output_directory)
