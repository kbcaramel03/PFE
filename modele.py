from ultralytics import YOLO
model = YOLO("yolov8n.pt")

results = model.train(
    data="thermal.yaml",
    imgsz=640,
    epochs=150,
    batch=16,      
    workers=4,
    device=0,
    patience=50,
    close_mosaic=10,
)
model = YOLO("runs/detect/train3/weights/best.pt")
metrics = model.val(data="thermal.yaml", imgsz=640)
print(metrics)
model.predict(
    source="dataset_mix/images/val",
    imgsz=640,
    conf=0.15,
    save=True
)
