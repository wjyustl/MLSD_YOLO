import math
import os
import numpy as np
import cv2
import torch
from torch.nn import functional as F


def decode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
    '''
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    '''
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1, )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def pred_lines(image, model, input_shape=None, score_thr=0.10, dist_thr=20.0):
    if input_shape is None:
        input_shape = [512, 512]
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    # 图像缩放+添加alpha通道
    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA),
                                    np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    # 通道优先：[H, W, C]->[C, H, W]
    resized_image = resized_image.transpose((2, 0, 1))
    # 增加批次维度：->[1, C, H, W]
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    # 像素值归一化到[-1, 1]
    batch_image = (batch_image / 127.5) - 1.0
    # 转换为PyTorch张量并移到GPU上
    batch_image = torch.from_numpy(batch_image).float().cuda()
    # 取前3个通道RGB
    batch_image = batch_image[:, 0:3, :, :]
    # 将图像输入模型
    outputs = model(batch_image)
    # 模型输出进行解码，得到
    # pts: 中心点坐标（形状为[N, 2]，N为中心点数量）
    # pts_score: 每个中心点的置信度分数
    # vmap: 位移向量图（每个中心点对应4个位移值，分别表示起点x偏移、起点y偏移、终点x偏移、终点y偏移）
    pts, pts_score, vmap = decode_output_score_and_ptss(outputs, 1000, 3)
    # 从vmap中分离出起点位移（前两个通道）和终点位移（后两个通道），并计算每个中心点对应的线段长度（欧氏距离）
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))
    # 遍历每个中心点及其分数
    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
    # 线段坐标乘以2，从256x256的尺度转换到512x512
    lines = 2 * np.array(segments_list)
    # 映射回原始图像的尺寸
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines


def line_angle(line):
    """辅助函数：计算线段角度0-90-180"""
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return 0
    angle = math.atan2(dy, dx) * 180 / math.pi
    # angle = angle % 180  # 归一化到0-180度范围
    # if angle > 90 and cate == "90":
    #     angle = 180 - angle
    # elif angle > 90 and cate == "180":
    #     angle = angle - 180
    return angle


def line_midpoint(line):
    """辅助函数：计算线段中点"""
    x1, y1, x2, y2 = line
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    return (mid_x, mid_y)


def rotate_img_and_lines(img, lines, angle, center):
    """旋转图像和线段"""
    # 旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算旋转后的图像边界，避免裁剪
    (h, w) = img.shape[:2]
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2
    # 执行旋转
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 将线段坐标转换到旋转后的空间
    rotated_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        point1 = np.array([x1, y1, 1], dtype=np.float32)
        point2 = np.array([x2, y2, 1], dtype=np.float32)
        rotated_point1 = M @ point1
        rotated_point2 = M @ point2
        rotatedx1, rotatedy1 = rotated_point1[:2]
        rotatedx2, rotatedy2 = rotated_point2[:2]

        rotated_lines.append([rotatedx1, rotatedy1, rotatedx2, rotatedy2])
    rotated_lines = np.array(rotated_lines)
    return M, rotated_img, rotated_lines

