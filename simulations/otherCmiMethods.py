import knncmi
import numpy as np
import pandas as pd
from scipy.special import digamma
from math import log

def countNeighborsCont(coord_dists, rho, coords = list()):
    '''
    input: list of coordinate distances (output of coordDistList), 
    coordinates we want (coords), distance (rho)

    output: scalar integer of number of points within ell infinity radius
    '''
    
    #ipdb.set_trace()
    if not coords:
        coords = range(coord_dists.shape[1])
    dists = np.max(coord_dists[:,coords], axis = 1)
    count = max(np.count_nonzero(dists < rho) - 1, 1)
    return count

def getKnnDistCont(distArray, k):
    '''
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value
    
    output: (k, distance to knn)
    '''
    dists = np.max(distArray, axis = 1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    return k, ordered_dists[k]

def getKnnDistRavk(distArray, k):
    '''
    input:
    distArray: numpy 2D array of pairwise, coordinate wise distances,
    output from getPairwiseDistArray
    k: nearest neighbor value
    
    output: (tilde k, distance to knn) tilde k = # pts < rho
    '''
    dists = np.max(distArray, axis = 1)
    ordered_dists = np.sort(dists)
    # using k, not k-1, here because this includes dist to self
    if ordered_dists[k] == 0:
        k_tilde = np.count_nonzero(dists == 0) - 1
    else:
        k_tilde = k
    return k_tilde, ordered_dists[k]

def cmiPointCont(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    cmi point estimate
    '''
    n = distArray.shape[1]
    coord_dists =knncmi.getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDistCont(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    nxz = countNeighborsCont(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighborsCont(coord_dists, rho, y_coords + z_coords)
    nz = countNeighborsCont(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    return xi

def miPointCont(point_i, x, y, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate
    '''
    n = distArray.shape[1]
    coord_dists = knncmi.getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDistCont(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    nx = countNeighborsCont(coord_dists, rho, x_coords)
    ny = countNeighborsCont(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
    return xi

def cmiCont(x, y, z, k, data, discrete_dist = 1):
    '''
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper param eter for kNN
    data: pandas dataframe

    output:
    scalar value of I(x,y|z)
    '''
    # compute CMI for I(x,y|z) using k-NN
    n, p = data.shape
    distArray = knncmi.getPairwiseDistArray(data, x + y + z, discrete_dist)
    if len(z) > 0:        
        s = 0
        for point in range(n):
            s = s + cmiPointCont(point, x, y, z, k, distArray)
        return(s/n)
    else:
        s = 0
        for point in range(n):
            s = s + miPointCont(point, x, y, k, distArray)
        return(s/n)  


def cmiPointR(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    cmi point estimate from Rahimzamani
    '''
    n = distArray.shape[1]
    coord_dists = knncmi.getPointCoordDists(distArray, point_i, x + y + z)
    k_tilde, rho = getKnnDistRavk(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    nxz = knncmi.countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = knncmi.countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = knncmi.countNeighbors(coord_dists, rho, z_coords)
    xi = digamma(k_tilde) - log(nxz + 1) - log(nyz + 1) + log(nz + 1)
    return xi

def miPointR(point_i, x, y, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    mi point estimate from Rahimzamani
    '''
    n = distArray.shape[1]
    coord_dists = knncmi.getPointCoordDists(distArray, point_i, x + y)
    k_tilde, rho = getKnnDistRavk(coord_dists, k)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    nx = knncmi.countNeighbors(coord_dists, rho, x_coords)
    ny = knncmi.countNeighbors(coord_dists, rho, y_coords)
    xi = digamma(k_tilde) + log(n) - log(nx + 1) - log(ny + 1)
    return xi

def cmiR(x, y, z, k, data, discrete_dist = 1):
    '''
    computes conditional mutual information, I(x,y|z)
    input:
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper parameter for kNN
    data: pandas dataframe

    output:
    scalar value of I(x,y|z) from Rahimzamani
    '''
    # compute CMI for I(x,y|z) using k-NN
    n, p = data.shape
    distArray = knncmi.getPairwiseDistArray(data, x + y + z, discrete_dist)
    if len(z) > 0:        
        s = 0
        for point in range(n):
            s = s + cmiPointR(point, x, y, z, k, distArray)
        return(s/n)
    else:
        s = 0
        for point in range(n):
            s = s + miPointR(point, x, y, k, distArray)
        return(s/n)
