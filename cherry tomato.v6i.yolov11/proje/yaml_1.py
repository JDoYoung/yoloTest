import yaml
import ultralytics
from ultralytics import YOLO

data = { 'train' : '/home/aa/cherry tomato.v6i.yolov11/train/images',
        'val' : '/home/aa/cherry tomato.v6i.yolov11/valid/images',
        'test' : '/home/aa/cherry tomato.v6i.yolov11/test/images',
        'names' : ['bug', 'level 1', 'level 2', 'level 3', 'level 4', 'level 5', 'level 6'],
        'nc' : 7 }

with open('/home/aa/cherry tomato.v6i.yolov11/data.yaml', 'w') as f:
    yaml.dump(data, f)

with open('/home/aa/cherry tomato.v6i.yolov11/data.yaml', 'r') as f:
    print(yaml.safe_load(f))


model = YOLO('yolo11n.pt')

print(type(model.names), len(model.names))

print(model.names)

model.train(data='/home/aa/cherry tomato.v6i.yolov11/data.yaml', epochs=30, patience=5, imgsz=416)

print(type(model.names), len(model.names))

print(model.names)