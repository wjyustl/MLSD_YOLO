from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO("yolov8n.pt")

    # Train the model
    # results = model.train(data="runs_yolo/corn_yolo.yaml", epochs=200, imgsz=512, batch=8, device=0, project="runs_yolo")
    # results = model.train(data="Project_Emergence/yolo.yaml", epochs=200, imgsz=512, batch=8, device=0, project="Project_Emergence")
    results = model.train(data="Project_Weed/weed_yolo.yaml", epochs=200, imgsz=512, batch=8, device=0, project="Project_Weed")
