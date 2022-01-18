import os 
import numpy as np
import cv2
import glob


true_label = []

path=".\\cifar10\\train"
path2=".\\cifar10\\test"

train_set = []
test_set = []

classes = []
classes_counts = []

#Read training images.
dataset_classes = glob.glob(".\\cifar10\\train" + "/*")
for folder in dataset_classes:
    path = folder.replace("\\", "/")
    if "/" in folder:
        class_name = folder.split("/")[-1]
    else:
        class_name = folder.split("\\")[-1]
        classes.append(class_name)
        train = glob.glob(path +"/*")
        #test = glob.glob(path2+ "/*")
        train_set.append(train)
        #test_set.append(test)
        classes_counts.append(0)

#Read testing images.
dataset_classes2 = glob.glob(".\\cifar10\\test" + "/*")
count = 0
for folder in dataset_classes2:
    path2 = folder.replace("\\", "/")
    if "/" in folder:
        class_name = folder.split("/")[-1]
    else:
        
        class_name = folder.split("\\")[-1]
        #train = glob.glob(path +"/*")
        test = glob.glob(path2+ "/*")
        #train_set.append(train)
        test_set.append(test)
        
        true_label.extend([count]*len(test))
        count+= 1
        #classes_counts.append(0)



#Function to calculate SIFT descriptors.
def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return des

# Descriptors per class.
def descriptors_from_class(class_img_paths, class_number):
    des = None
    print("descriptors_from_class started")
    
    for i in range(len(class_img_paths)):
        img_path = class_img_paths[i]
        img = cv2.imread(img_path)
        
        new_des = sift(img)
        if new_des is not None:
            if des is None:
                des = np.array(new_des, dtype=np.float32)
            else:
                des = np.vstack((des, np.array(new_des)))
    classes_counts[class_number] = len(des)
    print("descriptors_from_class done")
    return des

# Gets every local descriptor of a set with different classes (useful for getting a codebook).
def all_descriptors(class_list):
    des = None
    print("all_descriptors started")
    for i in range(len(class_list)):
        class_img_paths = class_list[i]
        new_des = descriptors_from_class(class_img_paths, i)
        if des is None:
            des = new_des
        else:
            des = np.vstack((des, new_des))
    print("all_descriptors done")
    return des



des = all_descriptors(train_set)

# Generate codebook.
def gen_codebook(descriptors, k = 64):
    iterations = 10
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    compactness, labels, centers = cv2.kmeans(descriptors, k, None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)
    return centers


codebook = gen_codebook(des, 200)


# Finds the nearest neighborhood for descriptor.
def find_nn(point, neighborhood):
    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point - neighbor)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i

    return nn, nn_idx


# Calculate the VLAD which is a global descriptor from a group of descriptors and centers that are codewords of a codebook, 
# obtained for example with K-Means.
def vlad(descriptors, centers):
    dimensions = len(descriptors[0])
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
    for descriptor in descriptors:
        nearest_center, center_idx = find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector, vlad_vector)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector


vlad_vector = vlad(des, codebook)

"""
	Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
    descriptor for each image and stores the label with the class of the image.
"""
def get_data_and_labels(img_set, codebook):
        y = []
        x = None
        for class_number in range(10):
            print("For class ", class_number)
            img_paths = img_set[class_number]
            print(len(img_paths))
            for i in range(len(img_paths)):
                print("Processing class", class_number)
                img = cv2.imread(img_paths[i])
                
                des = sift(img)
                if des is not None:
                    des = np.array(des, dtype=np.float32)
                    vlad_vector = vlad(des, codebook)
                    if x is None:
                        x = vlad_vector
                        y.append(class_number)
                    else:
                        x = np.vstack((x, vlad_vector))
                        y.append(class_number)
                
        y = np.float32(y)[:, np.newaxis]
        x = np.matrix(x)
        return x, y



x, y = get_data_and_labels(train_set, codebook)

from sklearn.svm import SVC
def train(train_labels, mega_histogram):
    clf  = SVC(kernel='poly')
    print("Training SVM")
    print(clf)
    print("Train labels", train_labels)
    clf.fit(mega_histogram, train_labels)
    print("Training completed")
    return clf



model = train(y.ravel(), x)


# Load test data.
test_x, test_y = get_data_and_labels(test_set, codebook)

result = []
for x in test_x:
    result.append(model.predict(x))


# Calculate predictions.
true = 0
class_wise = [0]*10
for i in range(10):
    #print("True label: ", true_label[i], "Prediction: ", result[i])
    true+= 1 if true_label[i] == int(result[i]) else 0
    class_wise[true_label[i]]+= 1 if true_label[i] == int(result[i]) else 0

for i in range(10):
    print(f'{classes[i]} true predictions: {class_wise[i]} total image:1000 Accuracy of {classes[i]} is {class_wise[i]/10}')

print("Acc.", true/len(true_label)*100)




