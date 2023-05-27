import cv2
import numpy as np


def traffic_light_recognition(img, x1, y1, x2, y2): # boundingbox -> list
    
    if x2-x1 >= y2-y1: return -1

    img = img[y1:y2, x1:x2,:]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    seg = int(hsv_img.shape[0]/2)
    top = np.sum(hsv_img[:seg,:,2])
    bot = np.sum(hsv_img[seg:,:,2])
    dif = top/bot
    # print("dif" ,dif)

    green_bound_lower = np.array([50, 20, 20])
    green_bound_upper = np.array([100, 255, 255])

    red_bound_lower = np.array([0, 20, 20])
    red_bound_upper = np.array([25, 255, 255])

    mask_green = cv2.inRange(hsv_img, green_bound_lower, green_bound_upper)
    kernel = np.ones((7,7),np.uint8)
    mask_red = cv2.inRange(hsv_img, red_bound_lower, red_bound_upper)


    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    
    green = np.sum(mask_green)
    red = np.sum(mask_red)

    if green == 0:
        color_dif = 1.2
    else:
        color_dif = red / green
    # print("color_dif",color_dif)

    if dif > 1.25 or dif < 0.8:
        if color_dif > 1:
            return 0
        else:
            return 1
    else:
        return -1

    # color_dif = red / green
    # # print("color_dif",color_dif)
    # if dif > 1.25 or dif < 0.8:
    #     if color_dif > 1:
    #         return 0 #red
    #     else:
    #         return 1 #green
    # else: 
    #     return -1    #not a pedes
    
    # if dif > 1.25 or color_dif > 5:
    #     return 0 # red
    # elif dif<0.8 or color_dif < 0.2:
    #     return 1 # green
    # else: 
    #     return -1 # not a pedes