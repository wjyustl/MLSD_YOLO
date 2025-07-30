import math
import os
import rasterio
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


def crop_window_tif(img_ori, img_path, if_save=False, window_size=640, stride=512):
    # img_pil = Image.open(input_path)
    # img_ori = cv2.imread(input_path)
    img = img_resize(img_ori, window_size, stride)
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
                save_mlsd_dir = data_dir + "_1CROP"
                os.makedirs(save_mlsd_dir, exist_ok=True)
                cv2.imwrite(os.path.join(
                    save_mlsd_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}_{j}.jpg"), crop)

            crop_list_row.append(crop)

        crop_list.append(crop_list_row)

    return crop_list, os.path.splitext(os.path.basename(img_path))[0]


def yolo_infer(model, img):
    results = model.predict(model="runs_yolo/corn_yolo.yaml", source=img, save=False, imgsz=512, conf=0.05, iou=0.2,
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


def merge_mask(img_ori, mask_list, if_save=None):
    img_mask = img_ori.copy()
    rows_mask = []
    for i in mask_list:
        row_mask = cv2.hconcat(i)
        rows_mask.append(row_mask)
    merged_mask = cv2.vconcat(rows_mask)

    mask = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
    h, w = mask.shape[:2]
    mh, mw = merged_mask.shape[:2]
    start_y = (h - mh) // 2
    start_x = (w - mw) // 2
    mask[start_y:start_y+mh, start_x:start_x+mw] = merged_mask
    img_mask = cv2.addWeighted(img_mask, 1, mask, 0.5, gamma=0)

    if if_save:
        save_res_dir = data_dir + "_6RES"
        os.makedirs(save_res_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_res_dir, f"{img_name}_img_mask.jpg"), img_mask)
        cv2.imwrite(os.path.join(save_res_dir, f"{img_name}_mask.png"), mask)
        if data_path.endswith(".tif"):
            replace_rgb_in_geotiff(data_path, img_mask,
                                   output_path=os.path.join(save_res_dir, f"{img_name}_img_mask.tif"))
    return img_mask, mask


def replace_rgb_in_geotiff(tif_path, opencv_im, output_path):
    # Step 1: 读取 GeoTIFF 并转为 (H, W, 4)
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        im_tif = src.read().transpose((1, 2, 0))  # (H, W, 4)
    # Step 2: 替换 RGB（opencv_im 是 BGR）
    rgb = cv2.cvtColor(opencv_im, cv2.COLOR_BGR2RGB)  # (H, W, 3)
    im_tif[..., 0:3] = rgb  # 替换 RGB
    # Step 3: 转回 rasterio 所需格式 (Bands, H, W)
    rgba = np.transpose(im_tif, (2, 0, 1))  # (4, H, W)
    # Step 4: 更新 profile 并写入
    profile.update({
        'count': 4,
        'dtype': rgba.dtype
    })
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgba)


if __name__ == '__main__':
    # TAG_SAVE
    TAG_SAVE_CROP_IMG = True
    TAG_SAVE_YOLO = True
    TAG_SAVE_RES = True

    # PARAMETERS
    WINDOWS_SIZE = 640
    CROP_STRIDE = 500
    IMG_SCOPE = (int((WINDOWS_SIZE - CROP_STRIDE) / 2), int((WINDOWS_SIZE + CROP_STRIDE) / 2))

    # CONFIG
    yolo_model_path = r"D:\PyProject\MLSD_YOLO\Project_Weed\merge_0730\weights\best.pt"
    data_path = r"D:\_DATA\Weed_Detection\changshu_weed_0729\changshu_zacao.tif"

    # START
    yolo_model = YOLO(yolo_model_path)
    data_dir = os.path.splitext(data_path)[0]

    restored_img_list = []
    restored_mask_list = []

    img_ori = cv2.imread(data_path)
    if img_ori.shape[2] != 3:
        img_ori = img_ori[:, :, :3]
    img_list, img_name = crop_window_tif(
        img_ori, data_path, if_save=TAG_SAVE_CROP_IMG, window_size=WINDOWS_SIZE, stride=CROP_STRIDE)

    for i, img_row in tqdm(enumerate(img_list)):
        restored_img_row_list = []
        restored_mask_row_list = []
        for j, img in enumerate(img_row):
            img_crop_name = f"{img_name}_{i}_{j}.jpg"

            yolo_img, mask_img = yolo_infer(yolo_model, img)

            if TAG_SAVE_YOLO:
                merged_img_mask = cv2.addWeighted(yolo_img, 1, mask_img, 0.5, gamma=0)
                save_line_dir = data_dir + "_2YOLO"
                os.makedirs(save_line_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_line_dir, img_crop_name), merged_img_mask)

            # 二维list保存掩码
            mask_img = mask_img[IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]]
            restored_mask_row_list.append(mask_img)
        restored_mask_list.append(restored_mask_row_list)

    # img_list = [[img[IMG_SCOPE[0]:IMG_SCOPE[1], IMG_SCOPE[0]:IMG_SCOPE[1]] for img in img_row] for img_row in img_list]
    # merged_img, merged_mask = merge_jpg_and_mask(img_list, restored_mask_list, if_save=TAG_SAVE_RES)
    img_ori, mask_ori = merge_mask(img_ori, restored_mask_list, if_save=TAG_SAVE_RES)
