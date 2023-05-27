
import numpy as np
from a_read_scans import get_lists_of_of_paths, fn_scan_to_array, fn_segm_mask_to_array
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans 
from numpy.linalg import norm
import cv2
import os


sift = cv2.xfeatures2d.SIFT_create()
subject_array = fn_scan_to_array("data/MRI_MASKS/subjects/9001104") # (384, 384, 160)
segm_mask = fn_segm_mask_to_array('9001104')                        # (384, 384, 160)
roi_mask = np.load("data/MRI_MASKS/roi_masks_dataset/roi_9001104.npy")


def makeshiftSIFT(array3d, coordinates):

    x, y, z = coordinates
    x,y,z=int(x),int(y),int(z)
    XYaxis=array3d[:,:,z].astype('uint8')
    XZaxis=array3d[:,y,:].astype('uint8')
    YZaxis=array3d[x,:,:].astype('uint8')

    keypoint1 = cv2.KeyPoint(x, y, 1)
    _, descriptor1 = sift.compute(XYaxis, [keypoint1] )

    keypoint2 = cv2.KeyPoint(x, z, 1)
    _, descriptor2 = sift.compute(XZaxis,[keypoint2])

    keypoint3 = cv2.KeyPoint(y, z, 1)
    _, descriptor3 = sift.compute(YZaxis,[keypoint3])

    descriptor = descriptor1 + descriptor2 + descriptor3 
    descriptor /= np.linalg.norm(descriptor)

    return descriptor


# descriptor = makeshiftSIFT(subject_array, np.array([20,160,30]))

# def sample_SVM_coordinates(roi_mask, segm_mask):

#     mask = roi_mask * segm_mask 
#     print(np.where(mask==1))

# sample_SVM_coordinates(roi_mask, segm_mask)
coordinates= np.load("data/temp/sampled_coordinates.npy")

def gather_local_descriptors(scan_array, list_of_coordinates):
    list_of_descriptors=[]
    for coordinate in list_of_coordinates:
        local_descriptor = makeshiftSIFT(scan_array, coordinate)
        list_of_descriptors.append(local_descriptor)
    local_descriptors_dataset = np.array(list_of_descriptors).reshape(len(list_of_descriptors), local_descriptor.shape[1])
    print("gathered local descriptors dataset.")
    return local_descriptors_dataset 

local_descriptor_dataset = gather_local_descriptors(subject_array, coordinates)

def image_specific_svm(dataset, segm_mask, coordinates):
    X = dataset
    print(X.shape)
    Y = []
    for coordinate in coordinates:
        x,y,z = coordinate
        Y.append(segm_mask[x,y,z])
    Y=np.array(Y)
    print(np.unique(Y))
    classifier = svm.LinearSVC(multi_class='ovr', random_state=1)
    classifier.fit(X,Y)
    W = classifier.coef_
    print(W.shape)
    print("Created W array")
    return W

W = image_specific_svm(local_descriptor_dataset, segm_mask, coordinates)
np.save('data/temp/W_sample.npy', W)

# print(W.shape)


coordinates_2 = np.array([[200,200,50],[210,142,128]])

def gather_assignment_dataset(coordinates):
    descriptors_dataset = np.empty((0, 128), dtype=np.float32)
    for scan in get_lists_of_of_paths("data/MRI_MASKS/subjects"):
        scan_array = fn_scan_to_array(scan)
        unq_scan_descriptors = gather_local_descriptors(scan_array, coordinates)
        descriptors_dataset =  np.vstack((descriptors_dataset, unq_scan_descriptors))
    print(descriptors_dataset.shape)
    return descriptors_dataset

# descriptors_dataset = gather_assignment_dataset(coordinates_2)
# np.save("data/temp/descriptors_dataset.npy", descriptors_dataset)


descriptors_dataset=np.load("data/temp/descriptors_dataset.npy")
def find_codewords(local_descriptors, no_of_words):
    random_state=1
    kmeans_model=KMeans(n_clusters=no_of_words, verbose=False, init='k-means++', random_state=random_state)
    kmeans_model.fit(local_descriptors)
    return kmeans_model.cluster_centers_

# codewords = find_codewords(descriptors_dataset, 20)
# print(codewords.shape)
# np.save("data/temp/codewords.npy", codewords)


# coordinates_per_subregion = ...
def soft_assign_histogram(codewords, array_3d, coordinates, a):

    subregion_descriptors = gather_local_descriptors(array_3d, coordinates)
    print(codewords.shape[0])
    histogram = np.zeros(codewords.shape[0])
    for descriptor in subregion_descriptors:
        for word in codewords:
            diff = np.linalg.norm(descriptor-word)**2
            c_1=np.exp(-a * diff)
            histogram+=c_1    
    
    return histogram

