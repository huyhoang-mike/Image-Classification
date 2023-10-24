from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data = r'C:\DISK (D)\HCMUT\6. CAREER\COMPUTER VISION\Image-Classification\weather-data', epochs=20, imgsz=64)

