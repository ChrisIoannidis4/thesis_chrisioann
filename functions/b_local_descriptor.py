import numpy as np
import cv2
from utils import getAverageMask, structure_tensor_3D, getGradients
from trimesh.creation import icosphere
from sklearn.utils.extmath import cartesian
from sklearn.metrics.pairwise import linear_kernel
from a_read_scans import fn_scan_to_array
import math
import itertools
from sklearn.preprocessing import normalize
from scipy.stats import kurtosis
from scipy.special import sph_harm
from scipy.ndimage.interpolation import map_coordinates


sift = cv2.xfeatures2d.SIFT_create()


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

# coordinate = np.load("data/temp/sampled_coordinates.npy")[10^6]

# descriptor_sift = makeshiftSIFT(fn_scan_to_array('data/MRI_MASKS/subjects/9005132'),coordinate)


# print('sift:', descriptor_sift.shape, np.max(descriptor_sift), np.min(descriptor_sift))



def hog_3d_proj(point, image, psize = 5, rsize = 15, orientation = 'vertices', level = 1, mode = 'aggregate', 
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
    projected_gradients /= gradient_magnitudes[:, :, None]

    # compute theshold to clip projected gradients and recalculate magnitude
    inner_prods = linear_kernel(axes)[0, :]
    thres = np.sort(inner_prods)[-2]

    projected_gradients -= thres
    projected_gradients[projected_gradients < 0] = 0
    projected_gradient_magnitudes = np.linalg.norm(projected_gradients, axis = 2)

    #distribute original magnitude in orientation bins
    gradient_histograms = projected_gradients * (gradient_magnitudes[:, :, None] / projected_gradient_magnitudes[:, :, None])
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

from skimage.feature import hog

def triplanar_hog(point, image, patch_size, n_bins, n_cells = 1, normalize = True):
    """
    Naive extension o Histogram of Oriented Gradients in 3D
    Calculating HOG descriptors for each plane intersecting the voxel (xy-, yz-, xz-)
    Final descriptor as a result of concatenation of planar descriptors.

    ...

    Parameters
    ----------
    point : tuple
        voxel cartesian coordinates
    image : memmap
        MRI containing voxel
    patch_size : int
        size of cubic patch around point for HOG extraction
    n_bins : int
        number of orientation histogram bins
    n_cells : int
        number of disjoint cubic cells to compartmentalize patch
    normalize : bool
        whether to normalize histogram norm or not

    Returns
    -------
    d : ndarray
        voxel HOG descriptor

    """

    x, y, z = point

    img_xy = image[:, :, z]
    img_yz = image[x, :, :]
    img_xz = image[:, y, :]

    step = patch_size // 2
    patch_xy = img_xy[x - step - 1 : x + step + 2,
                      y - step - 1 : y + step + 2]

    patch_yz = img_yz[y - step - 1 : y + step + 2,
                      z - step - 1 : z + step + 2]

    patch_xz = img_xz[x - step - 1 : x + step + 2,
                      z - step - 1 : z + step + 2]

    cell_size = patch_size // n_cells

    #encode each patch
    d_xy = hog(image = patch_xy, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)
    d_xz = hog(image = patch_xz, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)
    d_yz = hog(image = patch_yz, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)

    d = np.hstack((d_xy, d_yz, d_xz))

    if normalize:
        d = d / np.linalg.norm(d)

    return d




# descriptor_hog = hog_3d_proj(coordinate, fn_scan_to_array('data/MRI_MASKS/subjects/9005132'))

# print('hog:', descriptor_hog.shape, np.max(descriptor_hog), np.min(descriptor_hog))



def lbp_ri_sh(point, img, patch_size, sph_degree, ico_radius, ico_level, n_bins = 20, concatenate = True):
    '''
    Computes a 3D LBP texture descriptor for a region centered around a voxel. The intensity values 
    of the neighboring voxels is treated as a spherical function, and decomposed into a sum of
    spherical harmonics, achieving rotational invariance. A histogram of the texture codes is computed for
    each frequency (band) and the final descriptor is the concatenation of the above histograms.

    Reference : 3D LBP-Based Rotationally Invariant Region Description, 
                Banerjee J., Moelker A., Niessen W., Walsum  v. T., 
                ACCV 2012 Workshops, Part I, LNCS 7728, pp. 26-37, 2013

    ...

    Parameters
    ----------
    point : tuple
        the point to be described
    img : ndarray (width x height x depth)
        the MR image
    patch_size : int
        size of cellular patch around point
    sph_degree : int
        degree up to which to expand to spherical harmonics
    ico_radius : float
        radius of icosahedron to uniformly sample intensities around patch voxels
    ico_level : int
        controls refinement level of icosahedron
    n_bins : int
        number of bins for LBP histograms
    concatenate : bool
        if true, concatenate histograms, otherwise weighted aggregation

    Returns
    ------- 
    D : ndarray 
        LBP descriptor

    '''

    #extract image patch
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


# descriptor_lbp = lbp_ri_sh(coordinate, fn_scan_to_array('data/MRI_MASKS/subjects/9005132'), 5, 5, 3, 2, concatenate = False)

# print('lbp:', descriptor_lbp.shape, np.max(descriptor_lbp), np.min(descriptor_lbp))



def fn_hog_lbp_descriptor(coordinate, array):  
    descriptor_lbp = lbp_ri_sh(coordinate, array, 5, 5, 3, 2, concatenate = False)
    descriptor_hog = hog_3d_proj(coordinate, array) 
    descriptor_lbp_hog = np.concatenate([descriptor_hog,descriptor_lbp])
    return descriptor_lbp_hog.reshape(1, len(descriptor_lbp_hog))

# descriptor_hog_lbp = fn_hog_lbp_descriptor(coordinate, fn_scan_to_array('data/MRI_MASKS/subjects/9005132'))

# print('hog lbp:', descriptor_hog_lbp.shape, np.max(descriptor_hog_lbp), np.min(descriptor_hog_lbp))

# descriptor = makeshiftSIFT(fn_scan_to_array('data/MRI_MASKS/subjects/9005132'), coordinate)
# print(descriptor.shape)

'''
sift: (1, 128) 0.23224194 0.0
hog: (42,) 0.4880337225352944 0.0
lbp: (20,) 0.4356171635286237 0.05110432136505119
hog lbp: (1, 62) 0.4880337225352944 0.0
'''