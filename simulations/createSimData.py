import knncmi
import numpy as np
import pandas as pd
from scipy.special import digamma
import os
from otherCmiMethods import *
from simulations import *
import multiprocessing

def cmi4point(point_i, x, y, z, k, distArray):
    '''
    input:
    point_i: current observation row index
    x, y, z: list of indices
    k: positive integer scalar for k in knn
    distArray: output of getPairwiseDistArray

    output:
    tuple: (prop, ravk1, ravk2, fp), values of cmi point estimate for point i
    '''
    n = distArray.shape[1]
    coord_dists = knncmi.getPointCoordDists(distArray, point_i, x + y + z)
    
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))

    # FP calc
    k_tilde, rho = getKnnDistCont(coord_dists, k)
    nxz = countNeighborsCont(coord_dists, rho, x_coords + z_coords)
    nyz = countNeighborsCont(coord_dists, rho, y_coords + z_coords)
    nz = countNeighborsCont(coord_dists, rho, z_coords)
    xiFP = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    del k_tilde, nxz, nyz, nz

    # RAVK1 calc
    k_tilde, rho = getKnnDistRavk(coord_dists, k)
    nxz = knncmi.countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = knncmi.countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = knncmi.countNeighbors(coord_dists, rho, z_coords)
    xiRAVK1 = digamma(k_tilde) - log(nxz + 1) - log(nyz + 1) + log(nz + 1)
    del k_tilde, nxz, nyz, nz

    # RAVK2 calc
    k_tilde, rho = knncmi.getKnnDist(coord_dists, k)
    nxz = knncmi.countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = knncmi.countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = knncmi.countNeighbors(coord_dists, rho, z_coords)
    xiRAVK2 = digamma(k_tilde) - log(nxz + 1) - log(nyz + 1) + log(nz + 1)
    del k_tilde, nxz, nyz, nz
    
    # Prop calc
    k_tilde, rho = knncmi.getKnnDist(coord_dists, k)
    nxz = knncmi.countNeighbors(coord_dists, rho, x_coords + z_coords)
    nyz = knncmi.countNeighbors(coord_dists, rho, y_coords + z_coords)
    nz = knncmi.countNeighbors(coord_dists, rho, z_coords)
    xiProp = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)
    del k_tilde, nxz, nyz, nz
    
    return np.array([xiFP, xiRAVK1, xiRAVK2, xiProp])

def cmi4(x, y, z, k, data):
    '''
    Computes Prop, RAVK, FP CMI faster by not redoing computation
    x: list of indices for x
    y: list of indices for y
    z: list of indices for z
    k: hyper parameter for kNN
    data: pandas dataframe

    output:
    tuple: (fp, ravk1, ravk2, prop), values of I(x,y|z)
    '''
    n, p = data.shape
    if len(z) == 0:
        data['z'] = 0
        z = [p+1]
        
    distArray = knncmi.getPairwiseDistArray(data, x + y + z)
    ptEsts = map(lambda obs: cmi4point(obs, x, y, z, k, distArray), range(n))
    return sum(ptEsts)/n

def parallelSim(seed):
    runDat = pd.DataFrame({'seed':[], 'sampSize': [], 'dim': [], 'knn': [], 'title':[],
                       'data': [], 'FP': [], 'RAVK1':[], 'RAVK2':[], 'Proposed': [],
                       'trueInfo': []})
    np.random.seed(seed)
    for n in samp_sizes:
        for d in dims:
            for dat in dats:
                data, trueInfo, datName = dat(n,d)
                for k in ks:
                    z = list(range(2, 2+d))
                    cmiEst = cmi4([0], [1], z, k, data)
                    out_row = {'seed': seed, 'sampSize': n, 'dim': d, 'knn': k,
                               'title': datName, 'data': dat.__name__,
                               'FP': cmiEst[0], 'RAVK1': cmiEst[1],
                               'RAVK2': cmiEst[2], 'Proposed': cmiEst[3],
                               'trueInfo': trueInfo}
                    runDat = runDat.append(out_row, ignore_index = True)
    return(runDat)


if __name__ == '__main__':

    runs = 100  # Change to 100 to replicate simulations in paper
    samp_sizes = range(100, 1001, 100)
    dims = [1]
    dats = [cindep, corrUnif, mixture, discDep, contDep, contIndep, discIndep,
            contCondIndep, discCondIndep]
    ks = [7]

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    results = p.map(parallelSim, range(runs))    
    pd.concat(results).to_csv('expData.csv')
