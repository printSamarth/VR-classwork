import os 
import numpy as np
import cv2
import glob


#print(os.getcwd())

#Read images.
def getFiles(path=".\\cifar10\\train\\"):
    
    imlist = {} #image list
    count = 0
    for each in glob.glob(path + "*", recursive=True):
        #print(each)
        word = each.split("\\")[-1]
        print(" #### Reading image category ", word, " ##### ")
        imlist[word] = []
        for imagefile in glob.glob(path+word+"\\*"):
            #print("Reading file ", imagefile)
            im = cv2.imread(imagefile, 0)
            imlist[word].append(im)
            count +=1

    return [imlist, count]

# images: dict of image and  count: total count
images, count = getFiles() 


#Dictionary mapping between original label and encoded label.
name_dict = {}
train_labels = np.array([])
descriptor_list = []
label_count = 0 
sift_object = cv2.xfeatures2d.SIFT_create()


def features(image):
    keypoints, descriptors = sift_object.detectAndCompute(image, None)
    return [keypoints, descriptors]




#label_count: Doing transformation from word label to number label.
for word, imlist in images.items():
            name_dict[str(label_count)] = word
            print("Computing Features for ", word)
            for im in imlist:
                kp, des = features(im)
                
                #If there are no descriptor in image then don't use it.
                if des is None or des.any() == None:
                    continue
                descriptor_list.append(des)
                train_labels = np.append(train_labels, label_count)
            label_count += 1


#print(name_dict)

#print(np.shape(train_labels))
#print(descriptor_list)


# Cluster size.

K = 200
from sklearn.cluster import KMeans
kmeans_obj = KMeans(n_clusters = K,  verbose=1) #Let our vocabulary size be of 200 words.


# Function to stack all descriptors in shape of: M samples * N features
def formatND(l):
    print(np.shape(l[0]), np.shape(l[1]))
    vStack = np.array(l[0])
    print(len(l))
    count = 1
    for remaining in l[1:]:
        #print(np.shape(remaining))
        #print(count)
        count+= 1
        vStack = np.vstack((vStack, remaining))
    return vStack

# cluster using KMeans algorithm
def cluster(descriptor_vstack):    
    kmeans_ret = kmeans_obj.fit_predict(descriptor_vstack)
    return kmeans_ret

# Each cluster denotes a visual word, so encode each image in form of occurence of this visual words.
def developVocabulary(n_images, descriptor_list, kmeans_ret):

    mega_histogram = np.array([np.zeros(K) for i in range(n_images)])
    old_count = 0

    for i in range(len(descriptor_list)):
        l = len(descriptor_list[i])
        for j in range(l):
            idx = kmeans_ret[old_count+j] # We need old_count as an offset as we had passed stacked descriptor to kmeans.
            mega_histogram[i][idx] += 1
        old_count += l
    print("Vocabulary Histogram Generated")
    return mega_histogram



bov_descriptor_stack = formatND(descriptor_list)
#bov_descriptor_stack.shape

# It returns Index of the cluster each descriptor belongs to.
kmeans_ret = cluster(bov_descriptor_stack)
#len(kmeans_ret)
mega_histogram = developVocabulary(n_images = len(descriptor_list), descriptor_list = descriptor_list, kmeans_ret = kmeans_ret)

print(len(train_labels), len(mega_histogram))

#train_labels.shape
#mega_histogram.shape

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# Normalize the data
def standardize(mega_histogram):
    global scale
    scale = scale.fit(mega_histogram)
    mega_histogram = scale.transform(mega_histogram)
    return mega_histogram

mega_histogram = standardize(mega_histogram)


# Training function.

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train(train_labels, mega_histogram):
    clf  = SVC(kernel='poly')
    print("Training SVM")
    print(clf)
    print("Train labels", train_labels)
    clf.fit(mega_histogram, train_labels)
    print("Training completed")
    
    #model = KNeighborsClassifier(n_neighbors=4)
	#model.fit(mega_histogram, train_labels)
	#return model
    
    return clf

model = train(train_labels, mega_histogram)



### TESTING CODE.
testImages, testImageCount = getFiles(".\\cifar10\\test\\")

# Method to recognize a single image.
def recognize(test_img):

        kp, des = features(test_img)
        # print kp
        #print(des.shape)

        # generate vocab for test image
        vocab = np.array( [[ 0 for i in range(K)]])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        if des is None:
            return None
        test_ret = kmeans_obj.predict(des)
        
        #Generate BOW.
        for each in test_ret:
            vocab[0][each] += 1

        #print(vocab)
        # Scale the features
        vocab = scale.transform(vocab)

        # predict the class of the image
        lb = model.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb


# True count to count number of images classified correctly.
true_count = 0
total_count = 0
for word, imlist in testImages.items():
    #print("processing " ,word)

    # Class of current image.
    true_local = len(imlist)
    # Count of particular class images classified correctly.
    pre_local = 0

    for im in imlist:
        # print imlist[0].shape, imlist[1].shape
        #print(im.shape)
        cl = recognize(im)
        if cl is None:
            total_count+= 1
            continue
        #print(cl)
        
        true_count+= 1 if name_dict[str(int(cl[0]))] == word else 0
        total_count+= 1
        pre_local+= 1 if name_dict[str(int(cl[0]))] == word else 0
    print(f'{word} true predictions: {pre_local} total image: {true_local} Accuracy of {word} is {pre_local/true_local * 100}')




#Print overall acc.
print(true_count/total_count)
