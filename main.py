import cv2
import serial
import time
import argparse
import numpy as np
import torch

from numpy import random

from main_utils.make_box import convert_coor
# from main_utils.tracker import tracking

from yolov7.detect import detect
from yolov7.utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import select_device, time_synchronized


# SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/dataset/PTL_Dataset_876x657/heon_IMG_0575.JPG'
# SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/dataset/PTL_Dataset_876x657/heon_IMG_0521.JPG'
# WEIGHTS_CROSSWALK = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/epoch_029.pt' 
SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/crosswalk_vid.mp4'
WEIGHTS_CROSSWALK = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/epoch_029.pt'
WEIGHTS_SIGN_CAR = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/yolov7/yolov7x.pt'
IMG_SIZE = 640
DEVICE = ''   # cuda???
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


print("****************************************************************")
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

print()
print('Yolo Ready!!!')
print("****************************************************************")
print()

cur_frame = 0

if __name__ == '__main__':

    vids = cv2.VideoCapture(SOURCE)
    if not vids.isOpened():

        print("****************************************************************")
        print("Server not connected!!")
        print("****************************************************************")
        exit()

    while True:

        cur_frame += 1

        result, img0 = vids.read()
        # img0 = cv2.imread(SOURCE)  # BGR
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred_crosswalk = model_crosswalk(img, augment=False)[0]
            pred_sign_car = model_sign_car(img, augment=False)[0]

        pred_crosswalk = non_max_suppression(pred_crosswalk, conf_thres=0.10)
        pred_sign_car = non_max_suppression(pred_sign_car, conf_thres=0.25, classes=[2, 3, 5, 7, 9])
        #[0].cpu().numpy()

        if len(pred_crosswalk[0]) != 0:
            pred_crosswalk = convert_coor(img.shape[2:], pred_crosswalk, img0.shape)
        if len(pred_sign_car[0]) != 0:
            pred_sign_car = convert_coor(img.shape[2:], pred_sign_car, img0.shape)
        for idx in range(len(pred_crosswalk[0])):
            img2 = cv2.rectangle(img0, (int(pred_crosswalk[0][idx][0]), int(pred_crosswalk[0][idx][3])), (int(pred_crosswalk[0][idx][2]), int(pred_crosswalk[0][idx][1])), (0, 200, 0), 2)
        for idx in range(len(pred_sign_car[0])):
            if pred_sign_car[0][idx][5] != 9:
                img2 = cv2.rectangle(img0, (int(pred_sign_car[0][idx][0]), int(pred_sign_car[0][idx][3])), (int(pred_sign_car[0][idx][2]), int(pred_sign_car[0][idx][1])), (0, 0, 200), 2)
            else:
                img2 = cv2.rectangle(img0, (int(pred_sign_car[0][idx][0]), int(pred_sign_car[0][idx][3])), (int(pred_sign_car[0][idx][2]), int(pred_sign_car[0][idx][1])), (200, 0, 0), 2)
        
        cv2.imshow('frame', img2)

        if cv2.waitKey(100) == 27: 
            print('\n')
            print("Exit Program...")
            break

    vids.release()
    cv2.destroyAllWindows()