import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("Project_Weed/taibaicun_weed_det_0716/weights/best.pt")
img = cv2.imread(r"D:\PyProject\MLSD\datasets\taibaicun_weed_det\test\DJI_0951_5_2.jpg")

results = model.predict(model="runs_yolo/weed_yolo.yaml", source=img, save=False, imgsz=512, conf=0.5, iou=0.2, verbose=False)

print("原始类别ID:", results[0].boxes.cls)
print("模型类别映射:", results[0].names)
for i, box in enumerate(results[0].boxes):
    print(f"Box {i + 1}: conf={box.conf.item():.2f}, class={box.cls.item()}, xyxy={box.xyxy.tolist()}")

yolo_img = results[0].orig_img
mask_img = np.zeros_like(yolo_img)
for xyxy, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):

    xmin, ymin, xmax, ymax = map(int, xyxy)  # 将边界框坐标转换为整数
    if cls == 0:
        cv2.rectangle(mask_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), -1)
    else:
        cv2.rectangle(mask_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), -1)

yolo_img = cv2.addWeighted(yolo_img, 1, mask_img, 0.5, gamma=0)

# if yolo_img.shape[0] != mask_img.shape[0]:
#     mask_img = cv2.resize(mask_img, (int(mask_img.shape[1] * yolo_img.shape[0] / mask_img.shape[0]), yolo_img.shape[0]))
combined = np.hstack((img, yolo_img, mask_img))

cv2.imshow("combined", combined)
cv2.waitKey(0)
