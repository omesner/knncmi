import knncmi as knncmi
import numpy.random as rnd
import numpy as np
import pandas as pd
from scipy.special import digamma
import os
from contCMI import cmiCont
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from simulations import *

np.random.seed(59395)

expDat = pd.DataFrame({'sampSize': [], 'dim': [], 'knn': [], 'data': [], 
                     'Proposed': [], 'Continuous': [], 'trueInfo': []})

samp_sizes = range(100, 1001, 100)
dims = [1]
dats = [cindep, corrUnif, discDep, mixture]
ks = [7]


for n in samp_sizes:
    for d in dims:
        for dat in dats:
            data, trueInfo, datName = dat(n,d)
            for k in ks:
                z = list(range(2, 2+d))
                cmiEst = knncmi.cmi([0], [1], z, k, data)
                cmiContEst = cmiCont([0], [1], z, k, data)
                out_row = {'sampSize': n, 'dim': d, 'knn': k, 'title': datName, 'data': dat.__name__,
                               'Proposed': cmiEst, 'Continuous': cmiContEst, 'trueInfo': trueInfo}
                expDat = expDat.append(out_row, ignore_index = True)

#expDat.to_csv('contExpDat.csv')

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown', 'yellow']
    
grf = matplotlib.backends.backend_pdf.PdfPages("contPropComparison.pdf")
for dat in dats:
    fig = plt.figure()
    curDat = expDat.loc[expDat['data'] == dat.__name__]
    plt.plot('sampSize', 'Proposed', data=curDat, marker='o', markersize=4,
             color = 'b', linestyle = 'solid', linewidth=1,
             label = 'Proposed')
    plt.plot('sampSize', 'Continuous', data=curDat, marker='o', markersize=4,
             color = 'r', linestyle = 'solid', linewidth=1,
             label = 'Continuous')
    trueInfo = list(curDat['trueInfo'])[0]
    plt.axhline(y= trueInfo, color='black', linestyle='--', label = 'True CMI')
    #title = list(curDat['title'])[0]
    #plt.title(title)
    plt.legend()
    plt.xlabel("Sample Size")
    plt.ylabel("Estimated I(X;Y|Z)")
    grf.savefig(fig)
grf.close()
