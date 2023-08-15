
import numpy as np
import os
import SimpleITK as sitk
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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report


def get_lists_of_of_paths(directory):
    file_list = os.listdir(directory)
    file_paths = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)

    return file_paths  


def fn_scan_to_array(scan_no):
    mid_file=get_lists_of_of_paths('data/'+ scan_no)[0]
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

coordinate_cube = np.load('coordinates/coordinate_cube.npy')

def svm_coordinate_sampling(segm, roi):
    coord_labels = []
    coordinates = []
    
    mask_values = segm[coordinate_cube[:,0], coordinate_cube[:,1], coordinate_cube[:,2]]

    threshold_4 = np.count_nonzero(mask_values == 4) #.shape[0]#len(segm[segm[coordinate_cube]==4])
    
    threshold_1=int(threshold_4/1.3)
    threshold_2=int(threshold_4*1.3)
    threshold_3=int(threshold_4/1.3)
    threshold_0=int(threshold_4/1.4)
   
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
    '''
    Computes a 3D variant of the HOG Descriptor for an image region centered arounda voxel
    The Region of size (rsize x rsize x rsize) is compartmentalized into a set of disjoint patches,
    each of size (psize x psize x psize). A histogram of oriented gradients is computed for each patch, 
    with the orientation bins corresponding to vertices of centers of faces of a regular (refined) icosahedron.
    The final descriptor is a weighted aggregate of those histograms. Currently, implementation supports regions arranged in
    3x3x3 patches.

    Reference: Alexander Klaser, Marcin Marszalek, Cordelia Schmid. 
               A Spatio-Temporal Descriptor Based on 3D-Gradients. 
               BMVC 2008 - 19th British Machine Vision Conference, Sep 2008, Leeds, United Kingdom.pp.275:1-10. 
               DOI:10.5244/C.22.99

    ...

    Parameters
    ----------
    point : array - like
        the voxels to be characterized
    image : ndarray
        the image containing the voxels
    psize : int
        size of patches in region
    rsize : int
        size of region around central voxel
    orientation : string
        whether to associate histogram bins with vertices of centroids of faces of the icosahedron
    ico_coords : string
        coordinate system to define icosahedron on
    level : int
        number of refienement steps for icosahedron
    mode : string
        chooses whether to concatenate or aggregate patch histograms to form final descriptor
    
    Returns
    -------
    D : ndarray
        voxel descriptor

    '''

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
   

def fn_hog_lbp_descriptor(coordinate, array):  
    descriptor_lbp = lbp_ri_sh(coordinate, array, 5, 4, 2, 2, concatenate = True)
    descriptor_hog = hog_3d_proj(coordinate, array) 
    descriptor_lbp_hog = np.concatenate([descriptor_hog,descriptor_lbp])
    return descriptor_lbp_hog




def createtest_datasets(test_scan_no):
    
    scan = fn_scan_to_array(test_scan_no)
    mask = fn_segm_mask_to_array(test_scan_no)
    coord = svm_coordinate_sampling(mask, np.load('data/' +test_scan_no + '/roi.npy'))

    descriptors = []
    labels = []
    for coordinate in np.array(coord):
        descriptor = fn_hog_lbp_descriptor(coordinate, scan)
        # descriptor = np.reshape(descriptor, [descriptor.shape[1],])
        descriptors.append(descriptor)
        value = mask[coordinate[0], coordinate[1], coordinate[2]]
        if value == 0 or value == 1 or value == 3:
            labels.append(0)
        else :
            labels.append(1)   
        
        
    X = np.array(descriptors)
    y = np.array(labels)
    
    return X, y


