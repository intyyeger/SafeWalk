import os
import sys
sys.path.append(os.pardir)

from yolov7.utils.general import scale_coords


def convert_coor(img_shape, dets, im0_shape):

    for det in dets:

        if len(det):
            det[:, :4] = scale_coords(img_shape, det[:, :4], im0_shape).round()

    return dets


def draw_box():
    return 0