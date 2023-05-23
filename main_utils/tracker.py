import torch

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def ccwh_to_xywh(img_size, x,y,w,h):
    x_new = (x*img_size[1]) - (w*img_size[1] + 1) / 2
    y_new = (y*img_size[0]) - (h*img_size[0]+1) / 2
    return [x_new, y_new, w*img_size[1]+1, h*img_size[0]+1]


def yolo2byte(frame_idx, img_size, det):

    byte_annot = []
    for idx in range(len(det)):

        if det[idx][5] in [2, 3, 5, 7]:
            byte_annot.append([idx+1, -1]+ccwh_to_xywh(det[idx][0],det[idx][1], det[idx][2], det[idx][3])+[det[idx][4], det[idx][5], -1])
                
    return byte_annot


def tracking(frame_idx, img_size, det):

    converted_det = yolo2byte(frame_idx, img_size, det)
    converted_det = torch.tensor(converted_det)

    tracker = BYTETracker(BYTETrackerArgs())

    tracking = tracker.update(output_results=converted_det,
                                img_info=img_size, img_size=img_size)
    
    tracking_result = tracking.tlbr
    tracking_id = tracking.track_id

    return tracking_id, tracking_result
