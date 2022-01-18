import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


image = cv.imread("bb.jpg")
image = cv.resize(image, (700, 500))

#Do image smoothing.
bb = cv.bilateralFilter(image,15,75,75)

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(bb, kernel, iterations = 2)

gray = cv.cvtColor(erosion, cv.COLOR_BGR2GRAY) 
edged = cv.Canny(gray, 30, 100)  
#dilation = cv.dilate(edged, kernel, iterations = 1)
closing = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)

contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#Find arc area of all the contours than sort it and pick the length of array - 80th largest contours.
arc_area = []
for cnt in contours:
    arc_area.append(cv.contourArea(cnt))
arc_area.sort()

#variable length is threhold.
length = arc_area[len(arc_area)-80]

new_contours = []
for idx, cnt in enumerate(contours):
    if cv.arcLength(cnt, False) > length and idx < len(contours)-5:
        new_contours.append(cnt)
print('Number of books is: ', len(new_contours))

cv.drawContours(image, new_contours, -1, (0,255,0), 2)
cv.imshow('Contours', image)
cv.waitKey(0)







