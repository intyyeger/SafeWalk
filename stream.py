import cv2


SOURCE = 'C:/Users/J/Desktop/skku/skku_2023-1/SafeWalk/crosswalk_vid.mp4'


vids = cv2.VideoCapture(SOURCE)

while True:

    result, img0 = vids.read()

    cv2.imshow('frame', img0)

    if cv2.waitKey(300) == 27: break

vids.release()
cv2.destroyAllWindows()