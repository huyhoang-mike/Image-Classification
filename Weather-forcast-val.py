from ultralytics import YOLO
import numpy as np

# load a custom model
model = YOLO(r'C:\Users\Admin\runs\classify\train4\weights/best.pt') 

# Predict with the model
results = model('C:/DISK (D)/HCMUT/6. CAREER/COMPUTER VISION/Image-Classification\weather-data/val/Sunrise/sunrise1.jpg')  

# probs = results[0].probs.data
# print(np.argmax(probs))

print(results)

# np.argmax(probs).values()