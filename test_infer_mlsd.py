import torch
import os
import cv2
import math
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

from utils import pred_lines
from mlsd_pytorch.models.mbv2_mlsd_large import MobileV2_MLSD_Large

# mlsd_model_path = "runs_mlsd/2025-06-16_16-42-45/best.pth"
# input_path = r"D:\_DATA\DJI_0946_1_3.jpg"
mlsd_model_path = "runs_mlsd/2025-06-25_16-44-03/best.pth"
input_path = r"D:\_DATA\DJI_20250624185601_0004_D_4_1_4.jpg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileV2_MLSD_Large().cuda().eval()
model.load_state_dict(torch.load(mlsd_model_path, map_location=device), strict=True)

img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 置信度阈值0.1/线段长度阈值20
lines = pred_lines(img, model, [512, 512], 0.05, 20)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

mask = img.copy()
for line in lines:
    cv2.line(mask, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 200, 200), 1, 16)

cv2.imwrite(input_path.replace(".jpg", ".png"), mask)
