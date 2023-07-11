import numpy as np
from scipy.ndimage import prewitt, sobel, convolve1d



def getAverageMask(patch_size, mode = 'gaussian'):
    '''
    Generate a weighted average mask to cover the patch area 
    where a 3D-HOG descriptor is calculated
    ...

    Parameters
    ----------
    patch_size : int
        size of cubic area over which mask will be applied
    mode : string
        determines functional form of the distance encoded in mask

    Returns
    -------
    mask : ndarray
        mask array
    '''

    assert mode == 'gaussian' or mode == 'manhattan'

    ps = patch_size // 2

    if mode == 'gaussian':
        x, y, z = np.mgrid[-ps : ps + 1,
                           -ps : ps + 1,
                           -ps : ps + 1]
        
        mask = np.exp(-(x**2 / ps + y**2 / ps + z**2 / ps))
        mask /= mask.sum()
    
    else:
        mask_layers = []
        nrows = patch_size

        for layer in range(ps, -1, -1):
            add_exp = np.abs(np.arange(-ps, ps + 1)) + layer
            for count, nrow in enumerate(range(nrows)):
                if count == 0:
                    m = 2.**(nrows - (np.abs(np.arange(-ps, ps + 1)) + add_exp[count]))
                else:
                    m = np.vstack((m, 2.**(nrows - (np.abs(np.arange(-ps, ps + 1)) + add_exp[count]))))
            mask_layers.append(m)
        
        mask_layers_reverse = [mask_layers[- i - 1] for i in range(1, ps + 1)]
        mask = np.dstack((mask_layers + mask_layers_reverse))
        mask /= mask.sum()

    return mask


def structure_tensor_3D(grads, gmask):
    '''
    Computes structure tensor for a given region of a 3D image
    based on its partial derivatives, I= [Ix, Iy, Iz]
    The rotation matrix is extracted as a signed function over
    the eigenvectors of the structure tensor.
    A gaussian window is applied over the partial derivatives centered
    around the central voxel.

    ...

    Parameters
    ----------
    grads : ndarray (N, (2, 3))
        matrix of partial derivatives
    gmask : ndarray
        a gaussian window centered around the point of interest

    Returns
    -------
    R : ndarray (2, 2), (3, 3)
        Rotation matrix
    '''

    #compute structure tensor and perform eigendecomposition
    structure_tensor = lambda grads, gmask, i: gmask.ravel()[i] * np.array([[grads[i, 0]**2, grads[i, 0]*grads[i, 1], grads[i, 0]*grads[i, 2]],
                                 [grads[i, 1]*grads[i, 0], grads[i, 1]**2, grads[i, 1]*grads[i, 2]],
                                 [grads[i, 2]*grads[i, 0], grads[i, 2]*grads[i, 1], grads[i, 2]**2]])
    K = [structure_tensor(grads, gmask, i) for i in range(gmask.shape[0])]
    K = np.dstack(K).sum(axis = 2)   
    #K = (gmask.ravel()[:, None] * grads.reshape((-1, 3))).T.dot(grads.reshape((-1, 3)))
    d = gmask.ravel()[None, :].dot(grads.reshape((-1, 3)))
    _, Q = np.linalg.eigh(K)

    #get sign of product of directional derivatives with each eigenvector
    s = np.sign(d.dot(Q))

    #compute rotational matrix
    R = s * Q

    return R



def cartesianToSpherical(x, y, z):
    '''
    Convert from cartesian coordinate system to spherical ones. 
    The following convention is used:
        phi : azimuthal angle (longitude)
        theta : polar angle (lattitude)
    
    ...

    Parameters
    ----------
    x : ndarray
        x-coordinate
    y : ndarray
        y-coordinate
    z : ndarray
        z-coordinate

    Returns
    -------
    r : ndarray
        magnitude
    phi : ndarray
        azimuthal angle
    theta : ndarray
        polar angle
    '''
        
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.true_divide(z, r))

    return r, phi, theta


def getGradients(image, gradient_mask = 'base', coords = 'cart'):
    '''
    Calculate image gradients and return their corresponding values 
    expressed in spherical or cartesian coordinates
        
    ...    
    

    Parameters
    ----------
    image : ndarray
        image or image patch over which gradients will be computed
    gradient_mask : string
        type of mask to be used in convolution
    coords : string
        whether to return gradients in cartesian or spherical coordinates

    Returns
    -------
    (i_dx, i_dy, idz) : tuple of ndarrays
        partial derivatives (dx, dy, dz)
    or
    (m, phi, theta) : tuple of ndarrays
        gradient magnitude(r) and orientation(phi, theta) 
    '''

    if gradient_mask == 'prewitt':
        i_dx = prewitt(image, axis = 0)
        i_dy = prewitt(image, axis = 1)
        i_dz = prewitt(image, axis = 2)
    elif gradient_mask == 'sobel':
        i_dx = sobel(image, axis = 0)
        i_dy = sobel(image, axis = 1)
        i_dz = sobel(image, axis = 2)
    else:
        mask = np.array([1, 0, -1])     #Note:  Don't use [-1, 0, 1]! Remember mask is reversed in convolution
        i_dx = convolve1d(image, mask, axis = 0, mode = 'constant')[1:-1, 1:-1, 1:-1]
        i_dy = convolve1d(image, mask, axis = 1, mode = 'constant')[1:-1, 1:-1, 1:-1]
        i_dz = convolve1d(image, mask, axis = 2, mode = 'constant')[1:-1, 1:-1, 1:-1]

    if coords == 'sph':
        return cartesianToSpherical(i_dx, i_dy, i_dz)
    else:
        return i_dx, i_dy, i_dz
