import json
import os

# category_dict = {'weed': 0}  # 类别字典
category_dict = {'weed': 0}


def json_to_yolo(input_file_path, output_directory):
    """将LabelMe JSON标注转换为YOLO格式"""
    try:
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
                line = "%s %.4f %.4f %.4f %.4f\n" % (
                category_dict[shape['label']], cx, cy, width, height)  # 生成txt文件里每行的内容
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

    except Exception as e:
        print(f"Error processing {input_file_path}: {str(e)}")


def create_empty_yolo_file(image_path, output_directory):
    """为没有标注的图片创建空白YOLO标注文件"""
    try:
        # 获取图片文件名（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # 创建空白txt文件
        output_file_path = os.path.join(output_directory, f"{image_name}.txt")
        # 创建空文件
        open(output_file_path, 'w', encoding='utf-8').close()
        print(f"Created empty annotation for {image_name}")
    except Exception as e:
        print(f"Error creating empty annotation for {image_path}: {str(e)}")


def process_directory(input_directory, output_directory):
    """处理整个目录，为所有图片创建标注文件"""
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 支持的图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 收集所有图片文件和JSON文件
    all_files = os.listdir(input_directory)
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    json_files = [f for f in all_files if f.endswith('.json')]

    # 创建JSON文件名到图片文件名的映射
    json_to_image = {os.path.splitext(f)[0]: f for f in json_files}

    # 处理所有图片文件
    for img_file in image_files:
        # 获取图片文件名（不含扩展名）
        img_name = os.path.splitext(img_file)[0]

        # 查找对应的JSON文件
        if img_name in json_to_image:
            json_path = os.path.join(input_directory, json_to_image[img_name])
            json_to_yolo(json_path, output_directory)
        else:
            # 没有对应的JSON文件，创建空白标注
            img_path = os.path.join(input_directory, img_file)
            create_empty_yolo_file(img_path, output_directory)

    print(f"Processed {len(image_files)} images, created {len(image_files)} annotation files")


# 输入和输出目录
input_directory = r"D:\_DATA\Weed_Detection\_demo\data"
output_directory = r"D:\_DATA\Weed_Detection\_demo\label"

# 处理整个目录
process_directory(input_directory, output_directory)
