
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import cv2

sift = cv2.xfeatures2d.SIFT_create()



def get_lists_of_of_paths(directory):
    file_list = os.listdir(directory)
    file_paths = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)

    return file_paths  


def fn_scan_to_array(base_path):
    mid_file=get_lists_of_of_paths(base_path)[0]
    slice_list=get_lists_of_of_paths(mid_file)
    for i in range(160):
        if i==0:    
            slice = sitk.ReadImage(get_lists_of_of_paths(mid_file)[0])
            scan_array3D = (sitk.GetArrayFromImage(slice)).reshape(384,384)
        else:  
            slice = sitk.ReadImage(get_lists_of_of_paths(mid_file)[i])
            slice_array= sitk.GetArrayFromImage(slice).reshape(384,384)
            scan_array3D=np.dstack((scan_array3D, slice_array))
    return scan_array3D  # (384, 384, 160)

mri_scan = fn_scan_to_array('Baseline/KL4/9005132')



def fn_segm_mask_to_array(subject_name):

    mhd_path = "baseline/KL0/"+ subject_name +"/"+subject_name+".segmentation_masks.mhd"
    segm_mask = sitk.GetArrayFromImage(sitk.ReadImage(mhd_path, sitk.sitkFloat32))
    return np.flip(segm_mask, axis=0) # (384, 384, 160)


# def calculate_roi():

import os



# Specify the paths to the directories
# directory1 = 'data/MRI_MASKS/segmentation_masks'
# directory2 = 'data/MRI_MASKS/roi_masks_dataset'

# # Get the list of files in each directory
# segm = set([os.listdir(directory1)[i].split('.')[0] for i in range(len(os.listdir(directory1)))])
# roi = set([os.listdir(directory2)[i].split('_')[1][:7]for i in range(len(os.listdir(directory2)))])

# # Find the files in directory1 that are not in directory2
# files_only_in_directory = list(segm - roi)
# print('left exception join files loaded')
# Print the filenames
# for file in files_only_in_directory:
#     print(file)
'''
for subject_name in os.listdir("Baseline/KL4"):
    print('sj name: ', subject_name)
    mhd_path = "Baseline/KL4/"+ subject_name + "/" + subject_name +".segmentation_masks.mhd"
    segm_mask = sitk.ReadImage(mhd_path, sitk.sitkFloat32)
    print('segm mask read')
    binary_mask = np.logical_or(segm_mask == 2, segm_mask == 4).astype(np.uint8)#.reshape(1,386,386,160)
    print('binary_mask shape:', binary_mask.shape)
    binary_mask_with_channel = np.expand_dims(binary_mask, axis=0)
    print('binary mask ok', binary_mask_with_channel.shape)
    binary_image = sitk.GetImageFromArray(binary_mask_with_channel)
    dilate_filter = sitk.BinaryDilateImageFilter()
    print('filter init')
    kr=120
    dilate_filter.SetKernelRadius(kr)  # Adjust the radius according to your requirements
    print('filter ok - kernel radius: ', kr)
    # Perform the dilation operation
    dilated_image = dilate_filter.Execute(binary_image)

    # Convert the dilated image back to a numpy array
    dilated_mask = sitk.GetArrayFromImage(dilated_image)
    dilated_mask1 = np.reshape(dilated_mask, [384, 384, 160])
    print('dilated mask ok', dilated_mask1.shape)
    roi_mask = np.flip(dilated_mask1, axis=0)  # (384, 384, 160)
    print('roi_mask ok - kernel radius: ', kr)
    np.save("baseline_rois/roi_"+ subject_name +".npy" , roi_mask)
    print('~~~ DONE WITH', subject_name, '~~~')
 '''
# roi_mask = np.load("baseline_rois/roi_9003430.npy")
# # print(np.unique(roi_mask))
# subject_array  = fn_scan_to_array("Baseline/KL0/9003430") # (384, 384, 160)
# segm_mask = fn_segm_mask_to_array('9003430')                        # (384, 384, 160)
# # # print(np.unique(segm_mask))

# plt.imshow(subject_array[:, :, 50]  *roi_mask[:, :, 50])
# plt.colorbar()
# plt.show()
# plt.imshow(subject_array[:,:,50])#segm_mask[:, :, 50] * subject_array[:, :, 50])
# plt.colorbar()
# plt.show()
# plt.imshow(segm_mask[:, :, 50])#segm_mask[:, :, 50] * subject_array[:, :, 50])
# plt.colorbar()
# plt.show()
