import cv2
import numpy as np
import matplotlib.pyplot as plt

def createPanaroma(ig1, ig2, i1, i2):
    #Firstly, we have to find out the features matching in both the images
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(ig1,None)
    kp2, des2 = sift.detectAndCompute(ig2,None)
    
    #the obtained descriptors in one image are to be recognized in the other image too.
    #When we set parameter k=2, we are asking the knnMatcher to give out 2 best matches for each descriptor.
    #It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. 
    #And the closest one is returned.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    #So we filter out through all the matches to obtain the best ones. 
    #So we apply ratio test using the top 2 matches obtained above
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)
    
    
    #As you know that a homography matrix is needed to perform the transformation, 
    #and the homography matrix requires at least 4 matches, we do the following.
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(H)
    else:
        raise AssertionError("Can't find enough keypoints.")
        
    #we found the homography for transformation, we can now proceed to warp and stitch them together
    dst = cv2.warpPerspective(i1,H,(i2.shape[1] + i1.shape[1], i2.shape[0]))  
    plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
    plt.show()
    plt.figure()
    dst[0:i1.shape[0], 0:i2.shape[1]] = i2
    
    plt.imshow(dst)
    plt.show()
    return dst

def trimBlack(frame):
    #crop top
    if not np.sum(frame[0]):
        return trimBlack(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trimBlack(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trimBlack(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trimBlack(frame[:,:-2])    
    return frame

img_1 = cv2.imread('clg3.jpeg')
img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.imread('clg2.jpeg')
img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

img_3 = cv2.imread('clg1.jpeg')
img3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)

panaroma = createPanaroma(img2, img3, img_2, img_3) 
panaroma = trimBlack(panaroma)
plt.imshow(panaroma)
plt.show()
cv2.imshow("Intermediate", panaroma)
cv2.waitKey(0)

panaromaG = cv2.cvtColor(panaroma,cv2.COLOR_BGR2GRAY)
panaroma = createPanaroma(img1, panaromaG, img_1, panaroma) 
panaroma = trimBlack(panaroma)
plt.imshow(panaroma)
plt.show()
cv2.imshow("Final Panaroma", panaroma)
cv2.waitKey(0)



