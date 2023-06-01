import numpy as np
from a_read_scans import fn_scan_to_array, get_lists_of_of_paths, fn_segm_mask_to_array
import os
from sklearn.cluster import KMeans 
import random 

def load_roi(subj_no):
    roi = np.load("data/MRI_MASKS/roi_masks_dataset/roi_"+ subj_no + ".npy")
    return roi 


# path_to_subj = "data/MRI_MASKS/subjects/9001104"
# arr_1 = fn_scan_to_array(path_to_subj)
# subj_no = path_to_subj.split("/")[-1]
# roi_1 = load_roi(subj_no)
# segm_1 = fn_segm_mask_to_array(subj_no)
# print(segm_1.shape, path_to_subj.split("/")[-1])


def gather_roi_indices(roi_array):
    roi_indices=np.array(np.where(roi_array == 1)).T
    # print(roi_inices)
    roi_perimeter= [np.array(coord).tolist() for coord in roi_indices]
    return roi_perimeter
'''

dir1 = os.listdir("data/MRI_MASKS/subjects")
dir2 = [(os.listdir("data/MRI_MASKS/roi_masks_dataset")[i]).split(".")[0][-7:] for i in \
                    range(len(os.listdir("data/MRI_MASKS/roi_masks_dataset")))]

common_subjects = list(set(dir1).intersection(dir2)) 
'''
'''
['9017909', '9006723', '9036287', '9011115', '9057417', '9089627', '9034644', '9002817',
 '9056326', '9093622', '9036948', '9075815', '9041946', '9040944', '9039627', '9052335',
 '9090290', '9004175', '9073948', '9081306', '9044005', '9080864', '9027422', '9019287', 
 '9031426', '9011420', '9047539', '9005132', '9007827', '9083500', '9013161', '9001104',
 '9002430', '9033937', '9036770', '9069761']
'''
'''
print(common_subjects)
roi_perimeter = {}
for subject in common_subjects:
    roi_i=gather_roi_indices(load_roi(subject))
    print(len(roi_i))
    roi_perimeter = roi_perimeter.union(set(roi_i))
    print('ok', len(roi_perimeter))

roi_perimeter=np.array(roi_perimeter)
np.save('data/temp/coordinates_test1', roi_perimeter)

print('created perimeter')
sampled_coordinates= []

for coordinate in roi_perimeter:
    x,y,z = coordinate
    if x%6==0 and y%6==0 and z%6==0:
        sampled_coordinates.append([x,y,z])

np.save('data/temp/coordinates_test2.npy',sampled_coordinates)
print(sampled_coordinates.shape)


a_0, a_1, a_2, a_3, a_4 = 0, 0, 0, 0, 0
for coordinate in roi_perimeter:
    x,y,z=np.array(coordinate)
    if segm_1[x,y,z]==0 and x%15==0 and y%15==0 and z%15==0:
        sampled_coordinates.append(coordinate)
        a_0+=1
    elif segm_1[x,y,z]==1 and x%9==0 and y%9==0 and z%9==0:
        sampled_coordinates.append(coordinate)
        a_1+=1
    elif segm_1[x,y,z]==2 and x%4==0 and y%4==0 and z%4==0:
        sampled_coordinates.append(coordinate)
        a_2+=1
    elif segm_1[x,y,z]==3 and x%7==0 and y%7==0 and z%7==0:
        sampled_coordinates.append(coordinate)
        a_3+=1
    elif segm_1[x,y,z]==4 and x%3==0 and y%3==0 and z%3==0:
        sampled_coordinates.append(coordinate)
        a_4+=1
'''
# print(a_0, a_1, a_2, a_3, a_4)
# np.save("data/temp/sampled_coordinates", sampled_coordinates)

'''
import os
# print(fn_scan_to_array('data/MRI_MASKS/subjects/9001104').shape)
coordinates=[]
for x in range(20,320,5):
    for y in range(20,320,5):
        for z in range(10,150,5):
            # print(x%6==0 , y%6==0 , z%6==0)
            # if x%5==0 and y%5==0 and z%3==0:
            coordinates.append([x,y,z])
'''
# coordinates=np.array(coordinates)
# np.save('data/temp/general_coordinates.npy', coordinates)
# print(coordinates.shape)
#             coordinates.append([x,y,z])


# np.save('data/temp/coordinates_test1.npy', np.array(coordinates))

# coordinates = np.array(list(set(map(tuple, coordinates))))


# np.save('data/temp/coordinates_test1',coordinates)
# coordinates = np.load('data/temp/coordinates_test1.npy')
# print(len(coordinates))


# print(np.load('data/temp/coordinates_test1.npy').shape)


#GET ROI BOUNDARIES
'''
min_x = 1000
max_x = 0
min_y = 1000
max_y = 0
min_z = 1000
max_z = 0
for roi_path in get_lists_of_of_paths("data/MRI_MASKS/roi_masks_dataset"):
    print("path", roi_path)
    roi = np.load(roi_path)
    print("roi",roi.shape)
    coordinates = gather_roi_indices(roi)
    print("coord",np.array(coordinates).shape)
    print(coordinates[0])
    print(coordinates[0][0])
    x = np.array([coordinates[i][0] for i in range(len(coordinates))])
    y = np.array([coordinates[i][1] for i in range(len(coordinates))])
    z = np.array([coordinates[i][2] for i in range(len(coordinates))])

    roi_minx = np.min(x)
    roi_miny = np.min(y)
    roi_minz = np.min(z)

    roi_maxx = np.max(x)
    roi_maxy = np.max(y)
    roi_maxz = np.max(z)

    if min_x > roi_minx:
        min_x = roi_minx
    elif max_x < roi_maxx:
        max_x=roi_maxx
    if min_y > roi_miny:
        min_y = roi_miny
    elif max_y < roi_maxy:
        max_y=roi_maxy
    if min_z > roi_minz:
        min_z = roi_minz
    elif max_z < roi_maxz:
        max_z=roi_maxz


print("roi_minx", roi_minx)
print("roi_miny", roi_miny)
print("roi_minz", roi_minz)
print("roi_maxx", roi_maxx)
print("roi_maxy", roi_maxy)
print("roi_maxz", roi_maxz)
'''

# roi_minx 117
# roi_miny 87
# roi_minz 0
# roi_maxx 240
# roi_maxy 275
# roi_maxz 159

roi_cube = []

for x in range(117, 240):
    for y in range(87, 275):
        for z in range(0,160):
            roi_cube.append([x,y,z])

# np.save("data/temp/roi_cube.npy", roi_cube)

roi_cube = np.array(roi_cube)
random.shuffle(roi_cube)
random_state=1
kmeans_model1=KMeans(n_clusters=120, verbose=False, init='k-means++', random_state=random_state)
kmeans_model1.fit(roi_cube)
print("fit 1st clustering model")
kmeans_model2=KMeans(n_clusters=5, verbose=False, init='k-means++', random_state=random_state)
kmeans_model2.fit(kmeans_model1.cluster_centers_)
print("fit 2nd clustering model")
roi_subset = []
subregion_centers=kmeans_model2.cluster_centers_
for coordinate in roi_cube:
    x,y,z=coordinate
    if x%5==0 and y%5==0 and z%3==0:
        roi_subset.append(coordinate)
roi_subset=np.array(roi_subset)
print("roi_subset", roi_subset.shape)
first_label = kmeans_model1.predict(roi_subset)
final_label = kmeans_model2.predict(kmeans_model1.cluster_centers_[first_label])
print("creating subregions:")
subregion_1 = roi_subset[final_label==0]
subregion_2 = roi_subset[final_label==1]
subregion_3 = roi_subset[final_label==2]
subregion_4 = roi_subset[final_label==3]
subregion_5 = roi_subset[final_label==4]

print("     created 5 subregions:", subregion_1.shape, " -- ", subregion_2.shape, " -- ",subregion_3.shape,\
       " -- ",subregion_4.shape, " -- ",subregion_5.shape)
np.save("data/temp/sub_1_coord.npy", subregion_1)
np.save("data/temp/sub_2_coord.npy", subregion_2)
np.save("data/temp/sub_3_coord.npy", subregion_3)
np.save("data/temp/sub_4_coord.npy", subregion_4)
np.save("data/temp/sub_5_coord.npy", subregion_5)
