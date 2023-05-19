import cv2
import serial
import time
import argparse
import numpy as np
import torch

from numpy import random

from main_utils.make_box import convert_coor

from yolov7.detect import detect
from yolov7.utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device, time_synchronized



SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/dataset/PTL_Dataset_876x657/heon_IMG_0575.JPG'
WEIGHTS_CROSSWALK = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/epoch_029.pt'
WEIGHTS_SIGN_CAR = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/epoch_029.pt'
IMG_SIZE = 640
DEVICE = ''   # cuda???
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


print('YOLOv7 INITIALIZING...')
print()

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model_crosswalk = attempt_load(WEIGHTS_CROSSWALK, map_location=device)  # load FP32 model
model_sign_car = attempt_load(WEIGHTS_SIGN_CAR, map_location=device)    # load FP32 model
stride = int(model_crosswalk.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size

if half:
    model_crosswalk.half()  # to FP16
    model_sign_car.half()  # to FP16

# Get names and colors
names_crosswalk = model_crosswalk.module.names if hasattr(model_crosswalk, 'module') else model_crosswalk.names
names_sign_car = model_crosswalk.module.names if hasattr(model_crosswalk, 'module') else model_sign_car.names
colors_crosswalk = [[random.randint(0, 255) for _ in range(3)] for _ in names_crosswalk]
colors_sign_car = [[random.randint(0, 255) for _ in range(3)] for _ in names_sign_car]

# Run inference
if device.type != 'cpu':
    model_crosswalk(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_crosswalk.parameters())))  # run once
    model_sign_car(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_sign_car.parameters())))  # run once

print('Finished!!!')
print()

# def detect(frame):
#     # Load image
#     img0 = frame

#     # Padded resize
#     img = letterbox(img0, imgsz, stride=stride)[0]

#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)


#     # Inference
#     t0 = time_synchronized()
#     pred = model(img, augment=AUGMENT)[0]

#     # Apply NMS
#     pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

#     # Process detections
#     det = pred[0]

#     s = ''
#     s += '%gx%g ' % img.shape[2:]  # print string

#     if len(det):
#         # Rescale boxes from img_size to img0 size
#         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

#         # Print results
#         for c in det[:, -1].unique():
#             n = (det[:, -1] == c).sum()  # detections per class
#             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

#         # Write results
#         for *xyxy, conf, cls in reversed(det):
#             label = f'{names[int(cls)]} {conf:.2f}'
#             plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

#         print(f'Inferencing and Processing Done. ({time.time() - t0:.3f}s)')

#     # return results
#     return img0




if __name__ == '__main__':

    img0 = cv2.imread(SOURCE)  # BGR
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model_crosswalk(img, augment=False)[0]
        pred2 = model_sign_car(img, augment=False)[0]
    
    pred = non_max_suppression(pred, conf_thres=0.10)
    pred2 = non_max_suppression(pred2, conf_thres=0.05)
    #[0].cpu().numpy()

    print(pred)

    # 여기 차원 조심
    pred2 = convert_coor(img.shape[2:], pred2, img0.shape)
    print(pred2[0])
    print(img0.shape)
    img2 = cv2.rectangle(img0, (int(pred2[0][0][0]), int(pred2[0][0][3])), (int(pred2[0][0][2]), int(pred2[0][0][1])), (0, 0, 200), 3)
    cv2.imshow('r', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # while True:

    #     # 보행자가 정지한다면
    #     cap = cv2.VideoCapture(SOURCE)
    #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        




