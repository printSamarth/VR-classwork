import cv2 as cv
import numpy as np

img = cv.imread('face.jpg')
img = cv.resize(img, (700, 500))

# Find contour around face.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
edged = cv.Canny(gray, 70, 120)
kernel = np.ones((3,3),np.uint8)
closing = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
#cv.imshow('Edges', closing)
#cv.waitKey(0)
contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#Remove small contours which is actually a noise.
thres = 200
new_contours = []
for cnt in contours:
    if cv.contourArea(cnt) > thres:
            new_contours.append(cnt)


# Find position(cooridnates) of max area contour.
max_area = 0
n_x, n_y, n_w, n_h = 0, 0, 0, 0
for cnt in new_contours:
    x,y,w,h = cv.boundingRect(cnt)
    if w>50 and h>50 and w<200:
        if w*h >max_area:
            max_area = w*h
            n_x, n_y, n_w, n_h = x, y, w, h
            new_img = img[n_y:n_y+n_h,n_x:n_x+n_w] 

            
# Create mask image using above coordinates.
mask = np.zeros(img.shape[:2],np.uint8)
mask[n_y:n_y+n_h,n_x:n_x+n_w] = 255
res = cv.bitwise_and(img,img,mask = mask)

#Skin color range in HSV space.
min_HSV = np.array([0, 58, 30], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")
lower = np.array([0, 10, 60], dtype = "uint8") 
upper = np.array([20, 150, 255], dtype = "uint8")

#Find pixels belonging to this range and mark rest pixels as black.
imageHSV = cv.cvtColor(res, cv.COLOR_BGR2HSV)
skinRegionHSV = cv.inRange(imageHSV, lower, upper)
ma= np.zeros(img.shape[:2],np.uint8)
skinHSV = cv.bitwise_and(res, res, mask = skinRegionHSV)

# Finally mark all skin pixels as white.
gray = cv.cvtColor(skinHSV, cv.COLOR_BGR2GRAY) 
gray[np.where(gray != 0)] = 255

cv.imshow('Contours', gray)
cv.waitKey(0)

