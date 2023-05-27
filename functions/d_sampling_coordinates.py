import numpy as np
from a_read_scans import fn_scan_to_array, get_lists_of_of_paths, fn_segm_mask_to_array



def load_roi(subj_no):
    roi = np.load("data/MRI_MASKS/roi_masks_dataset/roi_"+ subj_no + ".npy")
    return roi 


path_to_subj = "data/MRI_MASKS/subjects/9001104"
arr_1 = fn_scan_to_array(path_to_subj)
subj_no = path_to_subj.split("/")[-1]
roi_1 = load_roi(subj_no)
segm_1 = fn_segm_mask_to_array(subj_no)
# print(segm_1.shape, path_to_subj.split("/")[-1])


def gather_roi_indices(roi_array):
    roi_inices=np.array(np.where(roi_1 == 1)).T
    # print(roi_inices)
    roi_perimeter= [np.array(coord) for coord in roi_inices]
    return roi_perimeter

roi_2= load_roi("9002430")
segm_2 = fn_segm_mask_to_array("9002430")
roi_perimeter=np.array(gather_roi_indices(roi_1))
print(roi_perimeter.shape)
# set_5 = set(map(tuple, gather_roi_indices(load_roi("9004175"))))

sampled_coordinates= []

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


print(a_0, a_1, a_2, a_3, a_4)
np.save("data/temp/sampled_coordinates", sampled_coordinates)