#  Method with real data
#  Author: Octavio Mesner

import pandas as pd
import numpy as np
import multiprocessing 
import os
import matplotlib.pyplot as plt
from functools import partial
import knncmi
from createSimData import cmi4point, cmi4

def import_raw_adult_data(rows = None):
    dat = pd.read_csv('../../compas-analysis/compas-scores.csv')
    out = dat[(dat['decile_score'] != -1) & (dat['is_recid'] != -1) &
              (abs(dat['days_b_screening_arrest']) < 90)][[
                  'race', 'decile_score', 'is_recid']]
    out['decile_score'] = out['decile_score']/10
    print(f"Data: column names: {list(out.columns)}, rows: {out.shape[0]}")
    return out

def create_sub_array(dat, var_list, size, seed):
    np.random.seed(seed)
    rand_obs = np.random.choice(dat.shape[0], size, replace = True)
    sub_dat = dat.iloc[rand_obs]
    distArray = knncmi.getPairwiseDistArray(sub_dat, var_list)
    return distArray

def get_outputs(size, array):
    f = partial(cmi4point, x=x, y=y, z=z, k=k, distArray=array)
    result = map(f, range(size))
    return np.mean(list(result), axis=0)
    
def run(seed):
    runDat = pd.DataFrame({'seed':[], 'sampSize': [], 'dim': [], 'knn': [], 'title':[],
                           'data': [], 'FP': [], 'RAVK1':[], 'RAVK2':[], 'Proposed': [],
                           'trueInfo': []})
    for samp_size in range(100, 1001, 100):
        dist_array = create_sub_array(dat, x+y+z, samp_size, seed)
        cmiEst = get_outputs(samp_size, dist_array)
        out_row = {
            'seed': seed, 'sampSize': samp_size, 'dim': 1, 'knn': k, 'title':
            'income data', 'data': 'income', 'FP': cmiEst[0], 'RAVK1': cmiEst[1], 'RAVK2':
            cmiEst[2], 'Proposed': cmiEst[3], 'trueInfo': None
        }
        runDat = runDat.append(out_row, ignore_index = True)
    print(f"run: {seed}")
    return runDat

def make_plot(dat):
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', 
              u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    sampSizes = dat['sampSize'].unique()
    fpDat = []
    ravk1Dat = []
    ravk2Dat = []
    propDat = []
    for size in sampSizes:
        plotDat = dat.loc[dat['sampSize'] == size]
        fpDat.append(plotDat['FP'].values)
        ravk1Dat.append(plotDat['RAVK1'].values)
        ravk2Dat.append(plotDat['RAVK2'].values)
        propDat.append(plotDat['Proposed'].values)

    plt.axhline(y= 0, color='black', linestyle='--')

    vp1 = plt.violinplot(fpDat, positions = sampSizes, widths = 50)
    vp2 = plt.violinplot(ravk1Dat, positions = sampSizes, widths = 50)
    vp3 = plt.violinplot(ravk2Dat, positions = sampSizes, widths = 50)
    vp4 = plt.violinplot(propDat, positions = sampSizes, widths = 50)

    plt.plot(sampSizes, dat.groupby(['sampSize']).median()['FP'].values,
             marker = 'x', color = colors[0], label = 'FP')
    plt.plot(sampSizes, dat.groupby(['sampSize']).median()['RAVK1'].values,
             marker = 'x', color = colors[1], label = 'RAVK1')
    plt.plot(sampSizes, dat.groupby(['sampSize']).median()['RAVK2'].values,
             marker = 'x', color = colors[2], label = 'RAVK2')
    plt.plot(sampSizes, dat.groupby(['sampSize']).median()['Proposed'].values,
             marker = 'x', color = colors[3], label = 'Proposed')
    plt.legend()
    plt.xlabel("Sample Size")
    plt.ylabel("Estimated I(X;Y|Z)")
    plt.savefig('race_recid.pdf')


if __name__ == '__main__':

    runs = 100
    k = 7
    x = [0] # race
    y = [1] # compas score 
    z = [2] # recidivism
    dat = import_raw_adult_data()
    p = multiprocessing.Pool(os.cpu_count())
    result =  p.map(run, range(runs))
    out = pd.concat(result)
    out.to_csv("recid.csv")
    make_plot(out)
    print("By race")
    pd.set_option('display.max_columns', 16)
    print(dat.groupby(['race']).describe())
    print("CMI ests on all data")
    tots = cmi4(x,y,z,k,dat)
    print(f"FP: {tots[0]}, RAVK1: {tots[1]}, RAVK2: {tots[2]}, Prop: {tots[3]}")
