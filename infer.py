import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm
from ultralytics import YOLO

from mlsd_pytorch.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import pred_lines, line_angle, rotate_img_and_lines


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileV2_MLSD_Large().cuda().eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    return model


def img_resize(img, window_size=640, stride=512):
    h, w = img.shape[:2]
    if h < window_size or w < window_size:
        raise ValueError(f'Window size {window_size} or {window_size} is too small')

    num_h = h // stride - (0 if h % stride > window_size - stride else 1)
    num_w = w // stride - (0 if w % stride > window_size - stride else 1)

    h_new = (num_h - 1) * stride + window_size
    w_new = (num_w - 1) * stride + window_size

    img = img[(h - h_new) // 2:(h + h_new) // 2, (w - w_new) // 2:(w + w_new) // 2, :]

    return img


def crop_window_tif(input_path, if_save=False, window_size=640, stride=512):
    img_pil = Image.open(input_path)
    img = cv2.imread(input_path)
    # tif-4通道 -> jpg-3通道
    if img.shape[2] != 3:
        img = img[:, :, :3]
    img = img_resize(img, window_size, stride)
    h, w = img.shape[:2]

    num_rows = (h - window_size + stride) // stride  # 行数
    num_cols = (w - window_size + stride) // stride  # 列数
    # 如果不能完全分割，则resize有误
    if (h - window_size) % stride != 0 or (w - window_size) % stride != 0:
        raise ValueError(f'Window size {window_size} or {window_size} is wrong')
    # 滑动窗口切割
    crop_list = []
    for i in range(num_rows):
        crop_list_row = []
        for j in range(num_cols):
            # 计算窗口坐标（防止越界）
            y1 = i * stride
            y2 = min(y1 + window_size, h)
            x1 = j * stride
            x2 = min(x1 + window_size, w)
            # 提取小图
            crop = img[y1:y2, x1:x2]
            # # 如果窗口不足 1000×1000，填充黑色背景
            # if crop.shape[0] < window_size or crop.shape[1] < window_size:
            #     pad_h = max(0, window_size - crop.shape[0])
            #     pad_w = max(0, window_size - crop.shape[1])
            #     crop = cv2.copyMakeBorder(
            #         crop, 0, pad_h, 0, pad_w,
            #         cv2.BORDER_CONSTANT, value=(0, 0, 0)
            #     )

            if if_save:
                save_mlsd_dir = data_dir + "_CROP"
                os.makedirs(save_mlsd_dir, exist_ok=True)
                cv2.imwrite(os.path.join(
                    save_mlsd_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}_{i}_{j}.jpg"), crop)

            crop_list_row.append(crop)

        crop_list.append(crop_list_row)

    return crop_list, os.path.splitext(os.path.basename(input_path))[0]


def mlsd_infer(model, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 置信度阈值0.1/线段长度阈值20
    lines = pred_lines(img, model, [512, 512], 0.1, 20)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # if True:
    #     mask = img.copy()
    #     for line in lines:
    #         cv2.line(mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 200, 200), 1, 16)
    #     save_mlsd_dir = data_dir + "_MLSD"
    #     os.makedirs(save_mlsd_dir, exist_ok=True)
    #     cv2.imwrite(os.path.join(save_mlsd_dir, img_crop_name), mask)

    return lines


def filter_lines(img, lines, if_save=False):
    """根据主要角度方向过滤线段"""
    if len(lines) == 0:
        return lines

    # 计算所有线段的角度
    angles = [line_angle(line) for line in lines]

    """按角度聚类"""
    clustering = DBSCAN(eps=10, min_samples=2).fit(np.array(angles).reshape(-1, 1))
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    # 找出最大的簇（噪声簇label=-1除外）
    cluster_sizes = {}
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
        cluster_indices = np.where(labels == label)[0]
        # 只保留大小≥20的簇
        # if len(cluster_indices) >= 20:
        cluster_sizes[label] = len(cluster_indices)

    # 按簇大小排序（从大到小）
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)

    # 计算主簇的平均角度
    if sorted_clusters:
        # main_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
        main_cluster_label, _ = sorted_clusters[0]
        main_cluster_indices = np.where(labels == main_cluster_label)[0]
        # 计算主簇的平均角度
        main_angles = [angles[i] for i in main_cluster_indices]
        main_avg_angle = np.mean(main_angles) % 180
    else:
        main_avg_angle = None  # 没有主簇

    # 确保存在次簇
    sec_avg_angle = None
    # if len(sorted_clusters) > 1:
    #     sec_cluster_label, _ = sorted_clusters[1]
    #     sec_cluster_indices = np.where(labels == sec_cluster_label)[0]
    #     sec_angles = [angles[i] for i in sec_cluster_indices]
    #     sec_avg_angle = np.mean(sec_angles) % 180
    # else:
    #     sec_avg_angle = None

    filtered_lines_main = []
    filtered_lines_sec = []
    diff_list_main = []
    diff_list_sec = []
    for i, line in enumerate(lines):
        current_angle = angles[i]
        # 计算与主方向的差值
        diff_main = min(abs(current_angle - main_avg_angle), 180 - abs(current_angle - main_avg_angle))
        # 保留在主方向容忍范围内的线段
        if abs(diff_main) <= 5.0:
            filtered_lines_main.append(line)
            diff_list_main.append(diff_main)
        # 次方向
        if sec_avg_angle:
            diff_sec = min(abs(current_angle - sec_avg_angle), 180 - abs(current_angle - sec_avg_angle))
            if abs(diff_sec) <= 5.0:
                filtered_lines_sec.append(line)
                diff_list_sec.append(diff_main)

    if if_save:
        filter_mask = img.copy()
        for line in filtered_lines_main:
            cv2.line(filter_mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 200, 200), 1, 16)
        save_mlsd_dir = data_dir + "_MLSD_FILTER"
        os.makedirs(save_mlsd_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_mlsd_dir, img_crop_name), filter_mask)
        if sec_avg_angle:
            filter_mask_sec = img.copy()
            for line in filtered_lines_sec:
                cv2.line(filter_mask_sec, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 200, 200), 1,
                         16)
            cv2.imwrite(os.path.join(save_mlsd_dir, img_crop_name.replace(".jpg", "_sec.jpg")), filter_mask_sec)

    return np.array(filtered_lines_main), np.array(filtered_lines_sec), main_avg_angle, sec_avg_angle


def cluster_lines(rotated_img, rotated_lines, if_save1=False, if_save2=False):
    """聚类"""
    midpoints = (rotated_lines[:, 1] + rotated_lines[:, 3]) / 2
    clustering = DBSCAN(eps=5, min_samples=2).fit(midpoints.reshape(-1, 1))
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    rotated_mask = rotated_img.copy()
    crop_line_list = []
    for num, label in enumerate(unique_labels):
        if label == -1:  # 跳过噪声点
            continue
        # 获取当前聚类的线段
        cluster_lines = rotated_lines[labels == label]

        line, inlier_segments = fit_segments_to_line(cluster_lines)
        x1, y1, x2, y2 = line

        # 按照小图中的拟合线段旋转
        angle = line_angle(line)
        center = (x1 + x2) / 2, (y1 + y2) / 2

        M2, rotated_rotated_img, rotated_rotated_lines = rotate_img_and_lines(rotated_img, [line], angle, center)

        rotated_rotated_lines = rotated_rotated_lines[0]

        x1_rot, x2_rot = sorted([rotated_rotated_lines[0], rotated_rotated_lines[2]])
        y1_rot, y2_rot = sorted([rotated_rotated_lines[1], rotated_rotated_lines[3]])
        # 条形图宽度为检测的线段宽度
        # crop_line_img = rotated_rotated_img[int(y1_rot - 20): int(y2_rot + 20), int(x1_rot): int(x2_rot), :]
        # location = (int(x1_rot), int(y1_rot - 20), int(x2_rot), int(y2_rot + 20))
        # 条形图宽度为原图像宽度
        crop_line_img = rotated_rotated_img[int(y1_rot - 20): int(y2_rot + 20), :, :]
        location = (int(0), int(y1_rot - 20), int(rotated_rotated_img.shape[1]), int(y2_rot + 20))

        crop_line_list.append(
            {"M": M2, "rotated_rotated_img": rotated_rotated_img, "crop_line_img": crop_line_img, "location": location})

        if if_save1:
            save_line_dir = data_dir + "_LINE"
            os.makedirs(save_line_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_line_dir, img_crop_name.replace(".jpg", f"_{num}") + ".jpg"),
                        crop_line_img)

        cv2.line(rotated_mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 200), 1, 16)

    if if_save2:
        save_line_dir = data_dir + "_ROTATED"
        os.makedirs(save_line_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_line_dir, img_crop_name), rotated_mask)

    return crop_line_list


def fit_segments_to_line(segments, method='ransac', max_trials=100, residual_threshold=5.0):
    """
    将一组线段拟合为一条最佳直线

    param:
        segments: 线段列表，每个线段为[x1, y1, x2, y2]
        method: 拟合方法 - 'linear'（最小二乘）或'ransac'（鲁棒拟合）
        max_trials: RANSAC最大迭代次数
        residual_threshold: RANSAC残差阈值（像素）
    """
    if len(segments) == 0:
        return None, None, []

    # 1. 提取所有线段端点
    all_points = []
    for seg in segments:
        all_points.append([seg[0], seg[1]])  # 起点
        all_points.append([seg[2], seg[3]])  # 终点
    points = np.array(all_points)

    # 2. 使用选定的方法拟合直线
    if method == 'linear':
        # 最小二乘法拟合
        model = LinearRegression()
        model.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        # 获取直线参数 y = mx + c -> mx - y + c = 0
        m = model.coef_[0]
        c = model.intercept_
        a, b, c = m, -1, c

    else:  # RANSAC方法
        # 使用RANSAC鲁棒拟合
        ransac = RANSACRegressor(min_samples=0.5,  # 至少50%的点
                                 residual_threshold=residual_threshold,
                                 max_trials=max_trials)
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        # 获取内点索引
        inlier_mask = ransac.inlier_mask_

        # 提取内点并重新拟合（更精确）
        inlier_points = points[inlier_mask]
        if len(inlier_points) > 1:
            final_model = LinearRegression()
            final_model.fit(inlier_points[:, 0].reshape(-1, 1), inlier_points[:, 1])
            m = final_model.coef_[0]
            c = final_model.intercept_
            a, b, c = m, -1, c
        else:
            # 如果没有足够内点，使用最小二乘
            m = ransac.estimator_.coef_[0]
            c = ransac.estimator_.intercept_
            a, b, c = m, -1, c

    # 3. 规范化直线方程 (a, b, c)
    norm = math.sqrt(a ** 2 + b ** 2)
    a, b, c = a / norm, b / norm, c / norm

    # 4. 确定拟合直线的端点
    # 将所有点投影到直线上
    projections = []
    for point in points:
        # 点到直线的投影公式
        x, y = point
        t = -(a * x + b * y + c)
        proj_x = x + a * t
        proj_y = y + b * t
        projections.append([proj_x, proj_y])

    projections = np.array(projections)

    # 找到投影点中的最小和最大x值
    min_idx = np.argmin(projections[:, 0])
    max_idx = np.argmax(projections[:, 0])

    line = [
        projections[min_idx, 0], projections[min_idx, 1], projections[max_idx, 0], projections[max_idx, 1]
    ]
    # 5. 确定哪些原始线段属于内点
    inlier_segments = []
    for i, seg in enumerate(segments):
        # 计算线段中点到直线的距离
        mid_x = (seg[0] + seg[2]) / 2
        mid_y = (seg[1] + seg[3]) / 2
        distance = abs(a * mid_x + b * mid_y + c)

        if distance <= residual_threshold:
            inlier_segments.append(i)

    return line, inlier_segments


def yolo_infer(model, img):
    results = model.predict(model="runs_yolo/corn_yolo.yaml", source=img, save=False, imgsz=512, conf=0.1, iou=0.2,
                            verbose=False)

    # print("原始类别ID:", results[0].boxes.cls)
    # print("模型类别映射:", results[0].names)
    # for i, box in enumerate(results[0].boxes):
    #     print(f"Box {i + 1}: conf={box.conf.item():.2f}, class={box.cls.item()}, xyxy={box.xyxy.tolist()}")

    yolo_img = results[0].orig_img
    mask_img = np.zeros_like(yolo_img)
    for xyxy, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):

        xmin, ymin, xmax, ymax = map(int, xyxy)  # 将边界框坐标转换为整数
        if cls == 0:
            cv2.rectangle(mask_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), -1)
        else:
            cv2.rectangle(mask_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)

    yolo_img = cv2.addWeighted(yolo_img, 1, mask_img, 0.5, gamma=0)

    return yolo_img, mask_img


def restore_img(img, M, shape):
    # 计算逆矩阵
    inv_M = cv2.invertAffineTransform(M)
    # 原始尺寸
    w, h = shape[0], shape[1]
    # 计算逆旋转后的边界（可能需要裁剪）
    cos = np.abs(inv_M[0, 0])
    sin = np.abs(inv_M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # 调整逆矩阵的平移
    inv_M[0, 2] += (new_w - img.shape[1]) // 2
    inv_M[1, 2] += (new_h - img.shape[0]) // 2
    # 执行逆旋转
    restored_img = cv2.warpAffine(img, inv_M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))

    return restored_img


def filter_masks(filter_mask):
    # mask = np.zeros_like(filter_mask)
    # # 面积大小过滤
    red_mask = np.where((filter_mask[:, :, 2] > 200) & (filter_mask[:, :, 0] < 50) & (filter_mask[:, :, 1] < 50), 255, 0)
    # blue_mask = np.where((filter_mask[:, :, 0] > 200) & (filter_mask[:, :, 1] < 50) & (filter_mask[:, :, 2] < 50), 255, 0)
    red_contours, _ = cv2.findContours(red_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # blue_contours, _ = cv2.findContours(blue_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in red_contours:
        if cv2.contourArea(contour) < 1000:  # 使用>=更直观
            cv2.drawContours(filter_mask, [contour], -1, (255, 0, 0), -1)
    # for contour in blue_contours:
    #     if cv2.contourArea(contour) >= 10:
    #         cv2.drawContours(mask, [contour], -1, (255, 0, 0), -1)
    # filter_mask = mask

    # 形态学过滤
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # 开运算（先腐蚀后膨胀）消除小突出物
    filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_OPEN, kernel)
    # filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)

    return filter_mask


def merge_jpg_and_mask(img_list, mask_list, if_save=None):
    """opencv"""
    rows_jpg = []
    rows_mask = []
    for i, j in zip(img_list, mask_list):
        row_jpg = cv2.hconcat(i)
        row_mask = cv2.hconcat(j)
        rows_jpg.append(row_jpg)
        rows_mask.append(row_mask)
    merged_img = cv2.vconcat(rows_jpg)
    merged_mask = cv2.vconcat(rows_mask)

    merged_img_mask = cv2.addWeighted(merged_img, 1, merged_mask, 0.5, gamma=0)

    if if_save:
        save_res_dir = data_dir + "_RES"
        os.makedirs(save_res_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_res_dir, f"{img_name}_img.jpg"), merged_img)
        cv2.imwrite(os.path.join(save_res_dir, f"{img_name}_mask.png"), merged_mask)
        cv2.imwrite(os.path.join(save_res_dir, f"{img_name}_img_mask.png"), merged_img_mask)

    print("拼接完成！")
    return merged_img, merged_mask


if __name__ == '__main__':
    # TAG_SAVE
    TAG_SAVE_CROP_IMG = True
    TAG_SAVE_MLSD_MASK = True
    TAG_SAVE_ROTATED_IMG = True
    TAG_SAVE_LINE_IMG = True
    TAG_SAVE_YOLO_STRIP = True
    TAG_SAVE_YOLO_SQUARE = True
    TAG_SAVE_YOLO = True
    TAG_SAVE_RES = True

    # PARAMETERS
    WINDOWS_SIZE = 640
    CROP_STRIDE = 512
    IMG_SCOPE = (int((WINDOWS_SIZE - CROP_STRIDE) / 2), int((WINDOWS_SIZE + CROP_STRIDE) / 2))

    # CONFIG
    mlsd_model_path = r"D:\PyProject\MLSD\runs_mlsd\taihe_0625_hangdian_640\best.pth"
    yolo_model_path = "runs_yolo/taihe_corn_0707/weights/best.pt"
    data_path = r"D:\_DATA\taihe_0625\hangdian\TEST\DJI_20250625170747_0033.JPG"

    # START
    mlsd_model = load_model(mlsd_model_path)
    yolo_model = YOLO(yolo_model_path)
    data_dir = os.path.splitext(data_path)[0]

    restored_img_list = []
    restored_mask_list = []

    img_list, img_name = crop_window_tif(
        data_path, if_save=TAG_SAVE_CROP_IMG, window_size=WINDOWS_SIZE, stride=CROP_STRIDE)

    for i, img_row in tqdm(enumerate(img_list)):
        restored_img_row_list = []
        restored_mask_row_list = []
        for j, img in enumerate(img_row):
            img_crop_name = f"{img_name}_{i}_{j}.jpg"

            # mlsd算法推理
            lines = mlsd_infer(mlsd_model, img)

            # 线段分割结果的合并及过滤
            lines_main, lines_sec, main_avg_angle, sec_avg_angle = filter_lines(img, lines, if_save=TAG_SAVE_MLSD_MASK)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M1, rotated_img, rotated_lines = rotate_img_and_lines(img, lines_main, main_avg_angle, center)

            # todo：条形图检测
            crop_line_list = cluster_lines(
                rotated_img, rotated_lines, if_save1=TAG_SAVE_LINE_IMG, if_save2=TAG_SAVE_ROTATED_IMG)

            # 方形图的次方向
            # if len(lines_sec) > 0:
            #     M2, rotated_img2, rotated_lines2 = rotate_img_and_lines(img, lines_sec, sec_avg_angle, center)
            #     crop_line_list2 = cluster_lines(
            #     rotated_img2, rotated_lines2, if_save1=TAG_SAVE_LINE_IMG, if_save2=TAG_SAVE_ROTATED_IMG)
            # else:
            #     crop_line_list2 = []
            # crop_line_list_merge = crop_line_list + crop_line_list2

            rotated_mask = np.zeros_like(rotated_img)
            for crop_line in crop_line_list:
                try:
                    yolo_img, mask_img = yolo_infer(yolo_model, crop_line["crop_line_img"])
                except Exception as e:
                    mask_img = np.zeros_like(crop_line["crop_line_img"])

                rotated_rotated_mask = np.zeros_like(crop_line["rotated_rotated_img"])

                x1, y1, x2, y2 = crop_line["location"]
                rotated_rotated_mask[y1:y2, x1:x2, :] = mask_img

                restored_rotated_mask = restore_img(rotated_rotated_mask, crop_line["M"], shape=rotated_img.shape[:2])

                rotated_mask += restored_rotated_mask
            # 重复检测的区域
            rotated_mask[:, :, 2] = np.where(
                (rotated_mask[:, :, 0] > 0) & (rotated_mask[:, :, 2] > 0), 0, rotated_mask[:, :, 2])
            restored_mask = restore_img(
                rotated_mask, M1, shape=img.shape[:2])[IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]]

            if TAG_SAVE_YOLO_STRIP:
                merged_img_mask = cv2.addWeighted(rotated_img, 1, rotated_mask, 0.5, gamma=0)
                save_line_dir = data_dir + "_YOLO_Strip"
                os.makedirs(save_line_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_line_dir, img_crop_name), merged_img_mask)

            # todo：方形图检测
            yolo_img2, mask_img2 = yolo_infer(yolo_model, rotated_img)
            restored_img2 = restore_img(yolo_img2, M1, shape=img.shape[:2])[
                             IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]]
            restored_mask2 = restore_img(mask_img2, M1, shape=img.shape[:2])[
                             IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]]

            if TAG_SAVE_YOLO_SQUARE:
                merged_img_mask = cv2.addWeighted(yolo_img2, 1, mask_img2, 0.5, gamma=0)
                save_line_dir = data_dir + "_YOLO_Square"
                os.makedirs(save_line_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_line_dir, img_crop_name), merged_img_mask)

            # todo: merge条形图为底
            restored_mask_black = (restored_mask[:, :, 0] == 0) & (restored_mask[:, :, 1] == 0) & (
                        restored_mask[:, :, 2] == 0)
            restored_mask = np.where(restored_mask_black[:, :, np.newaxis], restored_mask2, restored_mask)

            # 掩码过滤
            restored_mask = filter_masks(restored_mask)

            if TAG_SAVE_YOLO:
                merged_img_mask = cv2.addWeighted(img[IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]], 1, restored_mask, 0.5, gamma=0)
                save_line_dir = data_dir + "_YOLO"
                os.makedirs(save_line_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_line_dir, img_crop_name), merged_img_mask)

            # 二维list保存掩码
            restored_mask_row_list.append(restored_mask)
        restored_mask_list.append(restored_mask_row_list)

    img_list = [[img[IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]] for img in img_row] for img_row in img_list]
    merged_img, merged_mask = merge_jpg_and_mask(img_list, restored_mask_list, if_save=TAG_SAVE_RES)
    # 计算缺苗率
    blue_area = np.count_nonzero(merged_mask[:, :, 0] == 255)
    red_area = np.count_nonzero(merged_mask[:, :, 2] == 255)
    print(f"缺苗率={'%.3f' % (red_area / (blue_area + red_area))}")
