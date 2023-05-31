import cv2


SOURCE = 'http://192.168.184.64:81/stream'
# SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/crosswalk_vid.mp4'
SAVE_PATH = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/vids/'

trial = 14
frame = 0

vids = cv2.VideoCapture(SOURCE)

while True:

    frame += 1

    result, img0 = vids.read()
    # img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if result == False: break

    save_path = SAVE_PATH + str(trial) + "/" + str(frame).zfill(5) + ".jpg"

    cv2.imwrite(save_path, img0)

    cv2.imshow('frame', img0)
    print("current frame: ", frame)
    print(save_path)

    if cv2.waitKey(100) == 27: break

vids.release()
cv2.destroyAllWindows()