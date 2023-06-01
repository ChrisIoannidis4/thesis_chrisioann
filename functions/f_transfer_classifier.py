import numpy as np
from b_get_XW_space import create_global_descriptor, makeshiftSIFT
from d_sampling_coordinates import load_roi, gather_roi_indices
from a_read_scans import fn_scan_to_array, fn_segm_mask_to_array


def transfer_weight_vector(xi, Dx, Dw, M): #, W_normalizer):
    c_1= Dx.T @ Dx + 0.6 * np.identity(25)
    L_xi= np.linalg.inv(c_1) @ Dx.T @ xi
    L_wi= M @ L_xi
    wi = Dw @ L_wi
    # wi *= W_normalizer
    # print("wi.shape " , wi.shape)
    # print(np.max(wi))
    return wi


def svm_predict(local_descriptor, weights, bias):
    scores = np.dot(local_descriptor, weights) + bias   # shape (5,)
    # return the class with the highest score
    return np.argmax(scores)

W = np.load("data/X_W_arrays/W_space.npy")
X = np.load("data/X_W_arrays/X_space.npy")
Dx = np.load("data/dictionaries/Dx.npy")
Dw = np.load("data/dictionaries/Dw.npy")
Lx = np.load("data/dictionaries/Lx.npy")
Lw = np.load("data/dictionaries/Lw.npy")
M = np.load("data/dictionaries/M.npy")


codewords=np.load("data/temp/codewords.npy")
scan=fn_scan_to_array("data/MRI_MASKS/subjects/9011115")
roi_perimeter = gather_roi_indices(load_roi("9011115"))
subregion_list = [np.load(f"data/temp/sub_{i}_coord.npy") for i in range(1,6)]

test_voxel = roi_perimeter[1200]


print(test_voxel)
# print(subregion_list)
xi = create_global_descriptor(scan, subregion_list, codewords)
print(xi.shape)
wi_staging = transfer_weight_vector(xi, Dx, Dw, M)#, W_normalizer)
wi = wi_staging.reshape(128,5)


corr = 0
wrong = 0

for i in range (1,10000000,1000):
    test_voxel = roi_perimeter[i]
    x,y,z=test_voxel
    if int(fn_segm_mask_to_array('9011115')[x,y,z]) == 1:
        local_descriptor = makeshiftSIFT(scan, test_voxel)
        label= svm_predict(local_descriptor, wi)
        print(i,  label, fn_segm_mask_to_array('9011115')[x,y,z])

        if int(label)==int(fn_segm_mask_to_array('9011115')[x,y,z]):
            corr+=1
        else: 
            wrong+=1

print(corr/(corr+wrong))