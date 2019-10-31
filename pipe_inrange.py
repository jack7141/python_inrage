#-*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
# 자연광에 약함
low_H = 0
low_S = 0
low_V = 0
high_H = 179
high_S = 255
high_V = 255

def nothing(x):
    pass

cv.namedWindow('Canny')


cv.createTrackbar('low_H', "Canny" , low_H, 179, nothing)
cv.createTrackbar('high_H', "Canny" , high_H, 179, nothing)
cv.createTrackbar('low_S', "Canny" , low_S, 255, nothing)
cv.createTrackbar('high_S', "Canny" , high_S, 255, nothing)
cv.createTrackbar('low_V', "Canny" , low_V, 255, nothing)
cv.createTrackbar('high_V', "Canny" , high_V, 255, nothing)

cv.setTrackbarPos('low_H', 'Canny', 50)
cv.setTrackbarPos('high_H', 'Canny', 255)
cv.setTrackbarPos('low_S', 'Canny', 40)
cv.setTrackbarPos('high_S', 'Canny', 255)
cv.setTrackbarPos('low_V', 'Canny', 100)
cv.setTrackbarPos('high_V', 'Canny', 255)

img = cv.imread('/home/hgh/hgh/project/test_img/server_distance_img/etc/1.jpg')
img = cv.resize(img, dsize=(806, 605))
blurred = cv.GaussianBlur(img, (5, 5), 0)
kernel = np.ones((1,3), np.uint8)
kernel1 = np.ones((3,3), np.uint8)
while (1):

    low_H = cv.getTrackbarPos('low_H', 'Canny')
    high_H = cv.getTrackbarPos('high_H', 'Canny')
    low_S = cv.getTrackbarPos('low_S', 'Canny')
    high_S = cv.getTrackbarPos('high_S', 'Canny')
    low_V = cv.getTrackbarPos('low_V', 'Canny')
    high_V = cv.getTrackbarPos('high_V', 'Canny')
    frame_HSV = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # frame_threshold = cv.inRange(frame_HSV, (0, 68, 0), (197, 188, 255))
    # result = cv.erode(threshold, kernel, iterations = 5)
    result = cv.dilate(threshold, kernel1, iterations = 20)
    contours, _ = cv.findContours(result.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    img2 = img.copy()
    c = max(contours, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(c)
    roi_image = img2[y:y+h,x:x+w]

    roi_gray = cv.cvtColor(roi_image, cv.COLOR_BGR2GRAY)
    mask_inv = cv.bitwise_not(result)
    img2 = cv.bitwise_and(img2, img2, mask=result)

    cv.imshow('Canny', img2)
    cv.imwrite("inrange5.jpg",img2)
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()