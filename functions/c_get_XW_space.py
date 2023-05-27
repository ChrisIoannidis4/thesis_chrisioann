
import numpy as np
from a_read_scans import get_lists_of_of_paths, fn_scan_to_array, fn_segm_mask_to_array
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans 
from numpy.linalg import norm
import cv2
import os
from e_sampling_coordinates import load_roi
from b_local_descriptor import makeshiftSIFT, lbp_ri_sh, hog_3d_proj, fn_hog_lbp_descriptor, triplanar_hog

sift = cv2.xfeatures2d.SIFT_create()
# subject_array = fn_scan_to_array("data/MRI_MASKS/subjects/9001104") # (384, 384, 160)
# segm_mask = fn_segm_mask_to_array('9001104')                        # (384, 384, 160)
# roi_mask = np.load("data/MRI_MASKS/roi_masks_dataset/roi_9001104.npy")


# descriptor = makeshiftSIFT(subject_array, np.array([20,160,30]))

# def sample_SVM_coordinates(roi_mask, segm_mask):

#     mask = roi_mask * segm_mask 
#     print(np.where(mask==1))

# sample_SVM_coordinates(roi_mask, segm_mask)
# coordinates= np.load("data/temp/sampled_coordinates.npy")

def gather_local_descriptors(scan_array, list_of_coordinates):
    list_of_descriptors=[]
    specific_coordinates = []
    for coordinate in list_of_coordinates:
        local_descriptor =  makeshiftSIFT(scan_array, coordinate) #fn_hog_lbp_descriptor(coordinate, scan_array)
        list_of_descriptors.append(local_descriptor)
    local_descriptors_dataset = np.array(list_of_descriptors).reshape(len(list_of_descriptors), local_descriptor.shape[1])
    # print("gathered local descriptors dataset.
    return local_descriptors_dataset 
 


# local_descriptor_dataset = gather_local_descriptors(subject_array, coordinates)

def image_specific_svm(dataset, segm_mask, coordinates):
    X = dataset
    # print(X.shape)
    Y = []
    for coordinate in coordinates[:-20]:
        x,y,z = coordinate
        Y.append(segm_mask[x,y,z])
    Y=np.array(Y)
    # print(np.unique(Y))
    classifier = svm.LinearSVC(multi_class='ovr', random_state=1)
    classifier.fit(X,Y)
    W = classifier.coef_
    
    # print(W.shape)
    # print("Created W array")
    return W


import random

def svm_predict(local_descriptor, weights, bias):
    # print(local_descriptor.reshape(1, local_descriptor.shape[0]).shape, weights)
    scores = np.dot(weights, local_descriptor ) + bias  # shape (5,)
    # return the class with the highest score
    return np.argmax(scores)


def svm_test(dataset, segm_mask, coordinates):
    random.shuffle(coordinates)

    k = coordinates.shape[0]
    l= k//3
    X = dataset[:-l]
    # print(X.shape)
    Y = []
    for coordinate in coordinates[:-l]:
        x,y,z = coordinate
        Y.append(segm_mask[x,y,z])
    Y=np.array(Y)
    print("Y: ", np.unique(Y))
    # print(np.unique(Y))
    classifier = svm.LinearSVC(multi_class='ovr', random_state=1)
    classifier.fit(X,Y)
    W = classifier.coef_
    b= classifier.intercept_
    print("w, b, :" , W.shape, b.shape)
    #test_model
    x_test = dataset[-l:]
    y_test = []
    coordinates= coordinates[:-l]
    for coordinate in coordinates:
        x,y,z = coordinate
        y_test.append(segm_mask[x,y,z])
    y_pred = []
    corr, wrong = 0, 0

    for i, descriptor in enumerate(x_test):
        y_pred.append(classifier.predict(descriptor.reshape(1, descriptor.shape[0])))
        print("y pred, y act",classifier.predict(descriptor.reshape(1, descriptor.shape[0])), y_test[i])
        if classifier.predict(descriptor.reshape(1, descriptor.shape[0])) == y_test[i]:
            corr+=1
        else :
            wrong+=1

        

    print("acc:", str(100*corr/(wrong+corr)), "%")
    # print(W.shape)
    # print("Created W array")
    return W, b


coordinates= np.load("data/temp/general_coordinates.npy")
roi = load_roi("9002430")
segm_mask = fn_segm_mask_to_array("9002430")
roi_coord = []
for coordinate in coordinates:
    x,y,z=coordinate
    if roi[x,y,z] == 1:
        roi_coord.append(coordinate)
roi_coord = np.array(roi_coord)
print(roi_coord.shape)
array = fn_scan_to_array("data/MRI_MASKS/subjects/9002430")
dataset = gather_local_descriptors(array, roi_coord)
W, b = svm_test(dataset, segm_mask, roi_coord)

print(roi_coord.shape)

# segm_mask = fn_segm_mask_to_array("9002430")
# a = np.zeros(5)
# for coordinate in coordinates:
#     x,y,z = coordinate
#     segm_mask[x,y,z]
#     a[int(segm_mask[x,y,z])]+=1
# print(a)




'''
#GATHER WEIGHT SPACE
coordinates= np.load("data/temp/sampled_coordinates.npy")

dir1 = os.listdir("data/MRI_MASKS/subjects")
dir2 = [(os.listdir("data/MRI_MASKS/segmentation_masks")[i]).split(".")[0] for i in range(len(os.listdir("data/MRI_MASKS/segmentation_masks")))]
common_subjects = list(set(dir1).intersection(dir2))
print(common_subjects)
print(common_subjects)
Weight_Space=np.zeros([640,len(common_subjects)])
for i, subj_no in enumerate(common_subjects):


        print("scanning file: ",str(i),". ", subj_no)
        path = "data/MRI_MASKS/subjects/"+subj_no
        subject_array = fn_scan_to_array(path)
        segm_mask = fn_segm_mask_to_array(subj_no)  
        
        unq_image_local_descr=gather_local_descriptors(subject_array, coordinates)
        print("Created local descriptor dataset! Dimension: ", str(unq_image_local_descr.shape))
        Weight_Vector = image_specific_svm(unq_image_local_descr, segm_mask, coordinates).reshape(640,)
        print("Created Weight_Vector vector of size: ", Weight_Vector.shape)
        Weight_Space[:,i] = Weight_Vector
        print("Done")


np.save("data/X_W_arrays/W_space.npy", Weight_Space)



'''




# descriptors_dataset=np.load("data/temp/descriptors_dataset.npy")
def find_codewords(local_descriptors, no_of_words):
    random_state=1
    kmeans_model=KMeans(n_clusters=no_of_words, verbose=False, init='k-means++', random_state=random_state)
    kmeans_model.fit(local_descriptors)
    return kmeans_model.cluster_centers_

'''
# Gather All Local Descriptors Dataset and find codewords
coordinates_subset= np.load("data/temp/subeset_coord.npy")
descriptors_dataset = gather_assignment_dataset(coordinates_subset)
np.save("data/temp/descriptors_dataset.npy", descriptors_dataset)
descriptors_dataset= np.load("data/temp/descriptors_dataset.npy")
codewords = find_codewords(descriptors_dataset, 25)
print(codewords.shape)
np.save("data/temp/codewords.npy", codewords)
'''

# coordinates_per_subregion = ...
def soft_assign_histogram(codewords, array_3d, coordinates, a):

    subregion_descriptors = gather_local_descriptors(array_3d, coordinates)   #(2725, 128)
    # codewords (25, 128)
    histogram = np.zeros([1,codewords.shape[0]]) 
    for j, descriptor in enumerate(subregion_descriptors):
        descriptor_assignment=np.empty([1,codewords.shape[0]])
        for i, word in enumerate(codewords):
            diff = np.linalg.norm(descriptor-word)**2
            # c_1=np.exp(-a * diff)
            # print(c_1)
            descriptor_assignment[0][i] = np.exp(-a * diff)
        descriptor_assignment = descriptor_assignment/np.sum(descriptor_assignment)    
        histogram+=descriptor_assignment
    histogram/=np.sum(histogram)

    return histogram


# codewords=np.load("data/temp/codewords.npy")

# list_of_subregions = [np.load("data/temp/sub_1_coord.npy"),
#                       np.load("data/temp/sub_2_coord.npy"),
#                       np.load("data/temp/sub_3_coord.npy"),
#                       np.load("data/temp/sub_4_coord.npy"),
#                       np.load("data/temp/sub_5_coord.npy")]

def create_global_descriptor(scan, subregion_list, codewords):
    global_descriptor=np.empty([1,len(subregion_list)*codewords.shape[0]])
    for i, subregion in enumerate(subregion_list):
        subr_histogram =  soft_assign_histogram(codewords, scan, subregion, 0.8)  
        start_ind = i*codewords.shape[0]
        end_ind=i*codewords.shape[0]+codewords.shape[0]
        print(start_ind+1, "--", end_ind, "indices calculated")
        global_descriptor[0][start_ind:end_ind] = subr_histogram
    
    # print(global_descriptor)
    return global_descriptor.T

'''
dir1 = os.listdir("data/MRI_MASKS/subjects")
dir2 = [(os.listdir("data/MRI_MASKS/segmentation_masks")[i]).split(".")[0] for i in range(len(os.listdir("data/MRI_MASKS/segmentation_masks")))]
common_subjects = list(set(dir1).intersection(dir2))
print(common_subjects)
all_global_descriptors = np.empty([len(list_of_subregions)*codewords.shape[0], len(common_subjects)])
for i, subject in enumerate(common_subjects):
    path = "data/MRI_MASKS/subjects/" + subject
    scan_array = fn_scan_to_array(path)
    global_descriptor = create_global_descriptor(scan_array, list_of_subregions,  codewords)
    all_global_descriptors[:,i] = global_descriptor.reshape([len(list_of_subregions)*codewords.shape[0],])
    print("Done with ", subject)


np.save("data/X_W_arrays/X_space.npy", all_global_descriptors)
'''