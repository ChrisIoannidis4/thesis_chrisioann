from utils import getAverageMask, structure_tensor_3D, getGradients
from trimesh.creation import icosphere
from sklearn.utils.extmath import cartesian
from sklearn.metrics.pairwise import linear_kernel
import math
import itertools
from sklearn.preprocessing import normalize
from scipy.stats import kurtosis
from scipy.special import sph_harm
from scipy.ndimage.interpolation import map_coordinates
from sklearn.metrics import  classification_report
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
import SimpleITK as sitk



coordinate_cube = np.load('coordinates/coordinate_cube.npy')



def lbp_ri_sh(point, img, patch_size, sph_degree, ico_radius, ico_level, n_bins = 30, concatenate = True):
 
    psize = patch_size // 2
    x, y, z = point
    patch = img[x - psize - int(math.ceil(ico_radius)) : x + psize + int(math.ceil(ico_radius)) + 1,
                y - psize - int(math.ceil(ico_radius)) : y + psize + int(math.ceil(ico_radius)) + 1,
                z - psize - int(math.ceil(ico_radius)) : z + psize + int(math.ceil(ico_radius)) + 1]
    
    patch_coords = cartesian((range(patch_size), range(patch_size), range(patch_size))) + 1    

    #construct icosahedron for uniform sampling on sphere surface
    ico = icosphere(subdivisions = ico_level, radius = ico_radius)
    ico_coords = np.array(ico.vertices)
    theta = np.arccos(ico_coords[:, 2] / ico_radius)
    phi = np.arctan2(ico_coords[:, 1], ico_coords[:, 0])

    #get Spherical Harmonics expansion coefficients (up to degree sph_degree)
    m = list(itertools.chain.from_iterable([[i for i in range(-n,n+1)] for n in range(sph_degree)]))
    m = np.array(m)

    l = list(itertools.chain.from_iterable([[k for _ in range(2 * k + 1)] for k in range(sph_degree)]))
    l = np.array(l)

    Y = sph_harm(m[None, :], l[None, :], theta[:, None], phi[:, None])

    #sample sphere neighbors for each voxel in patch and interpolate intensity
    mapped_coords = patch_coords[None, :, :] + ico_coords[:, None, :]
    mapped_int = map_coordinates(patch, mapped_coords.T, order = 3)
    center_int = patch[ico_radius : -ico_radius, ico_radius : -ico_radius, ico_radius : -ico_radius]

    #Compute kurtosis (for better rotation invariance)
    kurt = kurtosis(mapped_int)

    #Apply sign function and pass obtain spherical expansion coefficients for each sample
    f = np.greater_equal(center_int.ravel()[:, None], mapped_int).astype('int')
    c = f.dot(Y)

    #obtain frequency components of threshold function by integrating and normalizing over orders m
    f = np.multiply(c[:, None, l == 0], Y[None, :, l == 0])
    for n in range(1, sph_degree):
        f = np.concatenate((f, np.sum(np.multiply(c[:, None, l == n], Y[None, :, l == n]),
                            axis=2, keepdims=True)), axis=2)
    f = np.sqrt(np.sum(f**2, axis=1))

    #keep real parts of decomposition and kurtosis
    f = np.real(f)
    kurt = np.real(kurt)

    #extract histograms
    H = np.histogram(kurt, bins = n_bins)[0]
    for i in range(sph_degree):
        H = np.column_stack((H, np.histogram(f[:, i], bins = n_bins)[0]))
    H = normalize(H, axis = 0)

    #Return Descriptor (concatenated or aggregated histograms)
    if concatenate is True:
        D = H.T.ravel()
    else: 
        D = H.sum(axis = 1)
    D = D / np.linalg.norm(D)

    return D





def hog_3d_proj(point, image, psize = 5, rsize = 15, orientation = 'vertices', level = 1, mode = 'concatenated', 
                rot_inv = False, norm = 'l2'):

    #sanity check
    assert type(rsize // psize) == int, print("Wrong combination of regional and patch sizes")

    #set params
    rs = rsize // 2
    ps = psize // 2
    ncells = rsize // psize
    
    # get icosahedron
    ico = icosphere(subdivisions = level)
    if orientation  == 'faces':
        axes = np.array(ico.face_normals)
    else:
        axes = np.array(ico.vertices)

    # get average masks
    region_mask = getAverageMask(rsize // psize, 'manhattan')
    patch_mask = getAverageMask(psize, 'manhattan')

    #calculate partial derivatives
    x, y, z = point 
    xp = range(- rs + ps, rs - ps + 1, psize)
    yp = range(- rs + ps, rs - ps + 1, psize)
    zp = range(- rs + ps, rs - ps + 1, psize)
    patch_centers = cartesian((xp, yp, zp))
    patch_locations = patch_centers + psize

    # extracting +1 voxel in each direction for computational consistency
    region = image[x - rs - 1 : x + rs + 2,
                   y - rs - 1 : y + rs + 2,
                   z - rs - 1 : z + rs + 2]
        
    i_dx, i_dy, i_dz = getGradients(region)

    #get gradients at the patch level
    dx = np.array([i_dx[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])

    dy = np.array([i_dy[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])
    
    dz = np.array([i_dz[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])

    dx = dx.reshape((ncells**3, psize**3))
    dy = dy.reshape((ncells**3, psize**3))
    dz = dz.reshape((ncells**3, psize**3))

    #collect all gradients in one array and calculate magnitudes
    raw_gradients = np.dstack((dx, dy, dz))
    if rot_inv is True:
        #rotate region according to dominant direction to achieve rotational invariance
        R = structure_tensor_3D(raw_gradients, getAverageMask(rsize, 'gaussian'))
        gradients = R.T.dot(raw_gradients.reshape((-1, 3)).T) 
        gradients = gradients.reshape(3, raw_gradients.shape[1], raw_gradients.shape[0]).T
    else:
        gradients = raw_gradients
    gradient_magnitudes = np.linalg.norm(gradients, axis = 2)

    #project gradients to icosahedron orientation axes
    projected_gradients = gradients.dot(axes.T)
    projected_gradients /= (gradient_magnitudes[:, :, None]+0.001)

    # compute theshold to clip projected gradients and recalculate magnitude
    inner_prods = linear_kernel(axes)[0, :]
    thres = np.sort(inner_prods)[-2]

    projected_gradients -= thres
    projected_gradients[projected_gradients < 0] = 0
    projected_gradient_magnitudes = np.linalg.norm(projected_gradients, axis = 2)

    #distribute original magnitude in orientation bins
    gradient_histograms = projected_gradients * (gradient_magnitudes[:, :, None] / (projected_gradient_magnitudes[:, :, None]+0.001))
    D = gradient_histograms.sum(axis = 1)

    if mode == 'flatten':
        Descriptor = (region_mask.ravel()[:, None] * D).ravel()
    else:
        Descriptor = region_mask.ravel().dot(D)

    if norm == 'l2':
        Descriptor = Descriptor / np.linalg.norm(Descriptor)
    if norm == 'l2-hys':
        Descriptor = Descriptor / np.linalg.norm(Descriptor)
        Descriptor = np.clip(Descriptor, a_min = 0, a_max = 0.25)
        Descriptor = Descriptor / np.linalg.norm(Descriptor)
    
    return Descriptor
   


def fn_hog_lbp_descriptor(coordinate, array, patch_size=5, sph_degree=3, ico_radius=2, ico_level=2, n_bins = 30, psize = 5, rsize = 15, level = 1):  
    descriptor_lbp = lbp_ri_sh(coordinate, array, patch_size=patch_size, sph_degree=sph_degree, ico_radius=ico_radius, ico_level=ico_level, n_bins = n_bins, concatenate = True)
    descriptor_hog = hog_3d_proj(coordinate, array, psize = psize, rsize = rsize, level = level) 
    descriptor_lbp_hog = np.concatenate([descriptor_hog,descriptor_lbp])
    return descriptor_lbp_hog.reshape(1, len(descriptor_lbp_hog))



def get_lists_of_of_paths(directory):
    file_list = os.listdir(directory)
    file_paths = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)

    return file_paths  

def svm_coordinate_sampling(segm, roi, ratio_0=1.4, ratio_1=1.3, ratio_2=1.3, ratio_3=1.3):
    coord_labels = []
    coordinates = []
    
    mask_values = segm[coordinate_cube[:,0], coordinate_cube[:,1], coordinate_cube[:,2]]

    threshold_4 = np.count_nonzero(mask_values == 4) #.shape[0]#len(segm[segm[coordinate_cube]==4])
    
    threshold_0=int(threshold_4/ratio_0)
    threshold_1=int(threshold_4/ratio_1)
    threshold_2=int(threshold_4*ratio_2)
    threshold_3=int(threshold_4/ratio_3)
   
    i_0,i_1,i_2,i_3,i_4 =0, 0, 0, 0, 0
    
    print(threshold_0,threshold_1,threshold_2,threshold_3,threshold_4)
        
        
    for coordinate in coordinate_cube:
        x,y,z = coordinate
        if roi[x,y,z]==1 and 360>x>6 and 360>y>6 and 150>z>6:
            if segm[x,y,z]==0 and i_0<threshold_0:
                coordinates.append(coordinate)
                i_0+=1
                coord_labels.append(0)
            elif segm[x,y,z]==1 and i_1<threshold_1:
                coordinates.append(coordinate)
                i_1+=1
                coord_labels.append(1)
            elif segm[x,y,z]==2 and i_2<threshold_2:
                coordinates.append(coordinate)
                i_2+=1
                coord_labels.append(2)
            elif segm[x,y,z]==3 and i_3<threshold_3:
                coordinates.append(coordinate)
                i_3+=1
                coord_labels.append(3)
            elif segm[x,y,z]==4 and i_4<threshold_4:
                coordinates.append(coordinate)
                i_4+=1 
                coord_labels.append(4)
                
    coord_labels=np.array(coord_labels)
    unique_values, counts = np.unique(coord_labels, return_counts=True)

    for value, count in zip(unique_values, counts):
        print(f"{value}: {count}")
    coordinates = np.array(coordinates)  
    np.random.shuffle(coordinates)      
    return coordinates


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



def fn_segm_mask_to_array(subject_name):

    mhd_path = "data/" + subject_name +"/"+subject_name+".segmentation_masks.mhd"
    segm_mask = sitk.GetArrayFromImage(sitk.ReadImage(mhd_path, sitk.sitkFloat32))
    return np.flip(segm_mask, axis=0) # (384, 384, 160)




mri_scan = fn_scan_to_array('data/9004175')
segm_mask = fn_segm_mask_to_array('9004175')
roi = np.load('data/9004175/roi.npy')



def svm_test(ratio_0=1.4, ratio_1=1.3, ratio_2=1.3, ratio_3=1.3, patch_size=5, sph_degree=3, ico_radius=2,\
                                            ico_level=2, n_bins = 30, psize = 5, rsize = 15, level = 1):
    
    coordinates = svm_coordinate_sampling(segm_mask, roi, ratio_0, ratio_1, ratio_2, ratio_3)
    descriptors = []
    labels = []
    for coord in coordinates:
        descriptor = fn_hog_lbp_descriptor(coord, mri_scan, patch_size, sph_degree, ico_radius, ico_level, n_bins,\
                                                                                psize, rsize, level)
        # if descriptor!= None # makeshiftSIFT(mri_scan, coord)
        descriptor = np.reshape(descriptor, [descriptor.shape[1],])
        descriptors.append(descriptor)
        value = segm_mask[coord[0], coord[1], coord[2]]
        if len(descriptors) == 1:
            print(descriptor.shape)
        labels.append(value)
        
        
    X = np.array(descriptors)
    y = np.array(labels)

    print('created datasets')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, y_train.shape )
    # Step 4: Train the SVM
    svm = SVC(kernel= 'linear', decision_function_shape ='ovr')
    print('fitting model')
    svm.fit(X_train, y_train)

    # Step 5: Predict on the test set
    y_pred = svm.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    test_report = classification_report(y_test, y_pred)
    print(test_report)
    
    return accuracy, test_report
    
    
    
    
grid_parameters = {
    "ratio_0" : [1.1, 1.4],
    "n_bins" : [25, 30, 40],
    "sph_degree" : [2, 3,4],
    "ico_radius" : [2, 3]    
    }

with open("test_report.txt", "a") as file:
    for ratio_0 in grid_parameters["ratio_0"]:
        for n_bins in grid_parameters["n_bins"]:
            for sph_degree in grid_parameters["sph_degree"]:
                for ico_radius in grid_parameters["ico_radius"]:
                    test_results = f"ratio_0: {ratio_0}, n_bins: {n_bins}, sph_degree: {sph_degree}, ico_radius: {ico_radius}\n\n"
                    try:
                        accuracy, test_report = svm_test(ratio_0=ratio_0, n_bins=n_bins, sph_degree=sph_degree, ico_radius=ico_radius)
                        test_results += f"Accuracy: {accuracy}\n {test_report}\n\n"
                    except:
                        test_results += "Test failed\n\n"
                        
                    test_results += "="*80+"\n\n"        
                    file.write(test_results)