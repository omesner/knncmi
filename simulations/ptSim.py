from knncmi import *
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simulations import *

np.random.seed(59395)

ptAn = pd.DataFrame({'sampSize': [], 'dim': [], 'knn': [], 'data': [], 'obs': [],
                     'cmiEst': [], 'ptEst': [], 'k_tilde': [], 'rho': [], 'nxz': [],
                     'nyz': [], 'nz': [], 'trueInfo': []})

samp_sizes = [1000]
dim = [1,5,9]
dat_sets = [contDep, contIndep, discDep, discDep]
ks = [2,6]


for n in samp_sizes:
    for d in dim:
        for dat in dat_sets:
            data, trueInfo = dat(n,d)
            for k in ks:
                z = list(range(2, 2+d))
                cmiEst = cmi([0], [1], z, k, data)
                distArray = getPairwiseDistArray(data)
                for itr in range(n):
                    pointDists = getPointCoordDists(distArray, itr)
                    k_tilde, rho = getKnnDist(pointDists, k)
                    nxz = countNeighbors(pointDists, rho, [0] + z)
                    nyz = countNeighbors(pointDists, rho, [1] + z)
                    nz = countNeighbors(pointDists, rho, z)
                    pt_est = cmiPoint(itr, [0], [1], z, k, distArray)
                    out_row = {'sampSize': n, 'dim': d, 'knn': k, 'data': dat.__name__,
                               'obs': itr, 'cmiEst': cmiEst, 'ptEst': pt_est,
                               'k_tilde': k_tilde, 'rho': rho, 'nxz': nxz, 'nyz': nyz,
                               'nz': nz, 'trueInfo': trueInfo}
                    ptAn = ptAn.append(out_row, ignore_index = True)

ptAn.to_csv('pointAnalysis.csv')
