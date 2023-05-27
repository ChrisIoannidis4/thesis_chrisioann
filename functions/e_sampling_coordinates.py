import numpy as np
from a_read_scans import fn_scan_to_array, get_lists_of_of_paths, fn_segm_mask_to_array
import os


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
    roi_inices=np.array(np.where(roi_array == 1)).T
    # print(roi_inices)
    roi_perimeter= [np.array(coord).tolist for coord in roi_inices]
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


import os
# print(fn_scan_to_array('data/MRI_MASKS/subjects/9001104').shape)
coordinates=[]
for x in range(20,320,5):
    for y in range(20,320,5):
        for z in range(10,150,5):
            # print(x%6==0 , y%6==0 , z%6==0)
            # if x%5==0 and y%5==0 and z%3==0:
            coordinates.append([x,y,z])

coordinates=np.array(coordinates)
np.save('data/temp/general_coordinates.npy', coordinates)
print(coordinates.shape)
#             coordinates.append([x,y,z])


# np.save('data/temp/coordinates_test1.npy', np.array(coordinates))

# coordinates = np.array(list(set(map(tuple, coordinates))))


# np.save('data/temp/coordinates_test1',coordinates)
# coordinates = np.load('data/temp/coordinates_test1.npy')
# print(len(coordinates))


# print(np.load('data/temp/coordinates_test1.npy').shape)


