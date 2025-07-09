import os
import shutil
from collections import defaultdict


def merge_image_files(source_folder, target_folder, copy_mode=True):
    """
    合并指定文件夹下所有图像文件到目标文件夹

    参数:
    source_folder: 要搜索的源文件夹路径
    target_folder: 合并后的目标文件夹路径
    copy_mode: True=复制文件(默认), False=移动文件
    """
    # 支持的图像文件扩展名
    image_extensions = ['.jpg', '.json']

    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)

    # 用于统计和重命名
    file_count = 0
    duplicate_count = 0
    name_counter = defaultdict(int)

    print(f"开始处理: {source_folder}")
    print(f"目标文件夹: {target_folder}")
    print(f"模式: {'复制' if copy_mode else '移动'}")

    # 遍历所有子文件夹
    for root, _, files in os.walk(source_folder):
        for filename in files:
            # 获取文件扩展名并转换为小写
            ext = os.path.splitext(filename)[1].lower()

            if ext in image_extensions:
                source_path = os.path.join(root, filename)

                # 处理文件名冲突
                # base_name = os.path.splitext(filename)[0]
                base_name = filename
                if name_counter[base_name] > 0:
                    new_filename = f"{base_name}_{name_counter[base_name]}{ext}"
                    duplicate_count += 1
                else:
                    new_filename = filename

                target_path = os.path.join(target_folder, new_filename)

                try:
                    # 执行复制或移动操作
                    if copy_mode:
                        shutil.copy2(source_path, target_path)  # 保留元数据
                    else:
                        shutil.move(source_path, target_path)

                    file_count += 1
                    name_counter[base_name] += 1

                    if file_count % 100 == 0:
                        print(f"已处理 {file_count} 个文件...")

                except Exception as e:
                    print(f"处理失败 {source_path} -> {target_path}: {str(e)}")

    print("\n处理完成!")
    print(f"共找到 {file_count} 个图像文件")
    print(f"重命名 {duplicate_count} 个重复文件名")
    print(f"目标文件夹现在包含 {len(os.listdir(target_folder))} 个文件")


if __name__ == "__main__":
    # 配置参数（根据需求修改）
    SOURCE_FOLDER = r"D:\_DATA\taihe_0625\hangdian\yolo"  # 替换为你的源文件夹
    TARGET_FOLDER = r"D:\_DATA\taihe_0625\hangdian\data"  # 目标文件夹名称

    # 执行合并（默认为复制模式）
    merge_image_files(SOURCE_FOLDER, TARGET_FOLDER, copy_mode=True)
