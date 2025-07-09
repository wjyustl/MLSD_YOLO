import os
import shutil
import argparse
from collections import defaultdict


def organize_images_into_batches(directory, batch_size=200):
    """
    将目录下的图像文件按批次组织到文件夹中

    参数:
        directory: 要遍历的根目录
        batch_size: 每批次的图像数量 (默认200)
    """
    # 支持的图像文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # 收集所有图像文件路径
    image_files = []

    # 遍历目录结构
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否是图像类型
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, file)
                image_files.append(full_path)

    total_images = len(image_files)
    if total_images == 0:
        print("未找到任何图像文件!")
        return

    # 创建输出目录
    output_dir = os.path.join(directory, "batched_images")
    os.makedirs(output_dir, exist_ok=True)

    # 将图像分组并移动到批次文件夹
    batch_count = (total_images + batch_size - 1) // batch_size  # 向上取整计算批次数量

    print(f"发现图像总数: {total_images}")
    print(f"创建批次数量: {batch_count}")
    print(f"每批图像数量: {batch_size} (最后一批可能较少)")
    print(f"输出目录: {output_dir}")

    for batch_index in range(batch_count):
        # 创建批次文件夹
        batch_dir = os.path.join(output_dir, f"batch_{batch_index + 1:03d}")
        os.makedirs(batch_dir, exist_ok=True)

        # 获取当前批次的图像
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size
        batch_files = image_files[start_idx:end_idx]

        print(f"\n处理批次 #{batch_index + 1:03d} ({len(batch_files)} 张图像) 到 {os.path.basename(batch_dir)}")

        # 移动图像到批次文件夹
        for i, src_path in enumerate(batch_files):
            filename = os.path.basename(src_path)
            dst_path = os.path.join(batch_dir, filename)

            # 处理文件名冲突
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                dst_path = os.path.join(batch_dir, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)
            print(f"  已移动: {filename} -> {os.path.basename(batch_dir)}/{os.path.basename(dst_path)}")

    print("\n处理完成! 所有图像已按批次组织。")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='将图像文件按批次组织到文件夹中')
    parser.add_argument('--directory', default=r"D:\_DATA\taihe_0625\hangdian\selected", help='包含图像的根目录路径')
    parser.add_argument('-b', '--batch-size', type=int, default=200,
                        help='每批次的图像数量 (默认: 200)')

    args = parser.parse_args()

    # 验证目录存在
    if not os.path.isdir(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在!")
        exit(1)

    # 执行批处理组织
    organize_images_into_batches(args.directory, args.batch_size)
