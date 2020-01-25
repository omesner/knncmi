import numpy.random as rnd
import numpy as np
import scipy as sp
import pandas as pd
import time as t
import os

def contDep(n,d, rho = 0.8):
    mean = [0,0]
    cov = [[1, rho], [rho, 1]]
    xycorr = rnd.multivariate_normal(mean, cov, n)
    zuncorr = rnd.multivariate_normal(np.zeros(d), np.identity(d), n)
    dat = pd.DataFrame(np.concatenate((xycorr, zuncorr), 1))
    info = (np.log(np.sqrt(1/(1 - rho ** 2))))
    name = 'Continuous and Dependent'
    return dat, info, name

def discDep(n,d):
    disc_dist = rnd.choice(list(range(4)), p = [0.4, 0.4, 0.1, 0.1], size = n)
    x = list()
    y = list()
    for itr in disc_dist:
        if itr == 0:
            x.append(1)
            y.append(1)
        elif itr == 1:
            x.append(-1)
            y.append(-1)
        elif itr == 2:
            x.append(1)
            y.append(-1)
        elif itr == 3:
            x.append(-1)
            y.append(1)
    dat = {'x': x, 'y': y}
    for itr in range(97, 97 + d):
        if itr % 2 == 0:
            dat[chr(itr)] = rnd.poisson(2,n)
        else:
            dat[chr(itr)] = rnd.binomial(3, 0.2, n)
    info = 2 * 0.4 * np.log(0.4/(0.5*0.5)) + 2 * 0.1 * np.log(0.1/(0.5*0.5))
    df = pd.DataFrame(data=dat)
    name = 'Discrete and Dependent'
    return df, info, name

def contIndep(n,d):
    dat = pd.DataFrame(rnd.multivariate_normal(np.zeros(d + 2), np.identity(d + 2), n))
    info = 0
    name = 'Continuous and Independent'
    return dat, info, name

def discIndep(n,d):
    dat = dict()
    for itr in range(97, 97 + d + 2):
        if itr % 2 == 0:
            dat[chr(itr)] = rnd.poisson(2,n)
        else:
            dat[chr(itr)] = rnd.binomial(3, 0.2, n)
    df = pd.DataFrame(data=dat)
    info = 0
    name = 'Discrete and Independent'
    return df, info, name

def contCondIndep(n,d):
    invcov = np.array([[1,0,0.5],[0,1,0.5],[0.5,0.5,1]])
    cov = np.linalg.inv(invcov)
    ciarray = rnd.multivariate_normal(np.zeros(3), cov, n)
    if d > 1:
        indarray = rnd.multivariate_normal(np.zeros(d - 1), np.identity(d - 1), n)
        dat = pd.DataFrame(np.concatenate((ciarray, indarray), 1))
    else:
        dat = pd.DataFrame(ciarray)
    info = 0
    name = 'Continuous and Conditionally Independent'
    return dat, info, name

def discCondIndep(n, d):
    unif = rnd.randint(10, size = n) + 1
    binom = rnd.binomial(unif, 0.5)
    pois = rnd.poisson(unif)
    dat = {'x': binom, 'y': pois, 'z': unif}
    if d > 1:
        for itr in range(97, 97 + d - 1):
            if itr % 2 == 0:
                dat[chr(itr)] = rnd.poisson(2,n)
            else:
                dat[chr(itr)] = rnd.binomial(3, 0.2, n)
    info = 0
    df = pd.DataFrame(data=dat)
    name = 'Discrete and Conditionally Independent'
    return df, info, name

# think about changing this to use the two functions above
def mixture(n,d):
    bern = rnd.binomial(n, 0.5, 1)
    mean = [0,0]
    rho = 0.8
    cov = [[1, rho], [rho, 1]]
    x, y = rnd.multivariate_normal(mean, cov, bern).T
    x = list(x)
    y = list(y)
    probs = [0.4, 0.4, 0.1, 0.1]
    disc_dist = rnd.choice(list(range(4)), p = probs, size = n - bern)
    for itr in disc_dist:
        if itr == 0:
            x.append(1)
            y.append(1)
        elif itr == 1:
            x.append(-1)
            y.append(-1)
        elif itr == 2:
            x.append(1)
            y.append(-1)
        elif itr == 3:
            x.append(-1)
            y.append(1)
    dat = {'x': x, 'y': y}
    for itr in range(97, 97 + d):
        if itr % 2 == 0:
            dat[chr(itr)] = rnd.poisson(2,n)
        else:
            dat[chr(itr)] = rnd.binomial(3, 0.2, n)
    df = pd.DataFrame(data=dat)
    info = 0.5 * 2 * 0.4 * np.log(2 * 0.4/(0.5*0.5))\
           + 0.5 * 2 * 0.1 * np.log(2 * 0.1/(0.5 * 0.5))\
           + 0.125 * np.log(4/(1 - rho ** 2))
    name = 'Observations Mixed and Dependent'
    return df, info, name

def corrUnif(n,d, m = 3):
    disc_unif = rnd.choice(list(range(m)), size = n)
    cont_unif = list()
    for itr in disc_unif:
        cont_unif.append(rnd.uniform(itr, itr + 2, 1)[0])
    dat = {'x': disc_unif, 'y': cont_unif}
    for itr in range(d):
        dat[str(itr)] = rnd.binomial(8,0.5,n)
    df = pd.DataFrame(data=dat)
    info = np.log(m) - (m - 1) * np.log(2)/m
    name = 'Variables Mixed and Dependent'
    return df, info, name

def mindep(n,d):
    m = 3
    disc_unif = rnd.choice(list(range(m)), size = n)
    cont_unif = rnd.uniform(0, m, size = n)
    dat = {'x': disc_unif, 'y': cont_unif}
    for itr in range(d):
        dat[str(itr)] = rnd.normal(0,1,n)
    df = pd.DataFrame(data=dat)
    info = 0
    name = 'Variables Mixed and Independent'
    return df, info, name

def cindep(n,d):
    beta = 10
    p = 0.5
    exp_dist = rnd.exponential(scale = beta, size = n)
    pois_dist = list()
    binom_dist = list()
    for itr in exp_dist:
        pois = rnd.poisson(itr, size = 1)[0]
        pois_dist.append(pois)
        binom_dist.append(rnd.binomial(pois, p, size = 1)[0])
    dat = {'exp': exp_dist, 'binom': binom_dist, 'pois': pois_dist}
    for itr in range(d-1):
        dat[str(itr)] = rnd.normal(0,1,n)
    df = pd.DataFrame(data=dat)
    info = 0
    name = 'Variables Mixed and Conditionally Independent'
    return df, info, name

def sim(filename, data_sets = [contDep, contIndep, discDep, discIndep, contCondIndep, discCondIndep]):
    samp_sizes = range(100, 1001, 100)
    dim = [1, 5, 9] # size on conditioning set
    ks = [3, 7]
    col_names = ['sampSize', 'dim', 'knn', 'data', 'runTime', 'cmiEst', 'truth', 'dataTitle']
    fname = filename + '.csv'
    out_file = open(fname, 'w+')
    out_file.write(','.join(col_names) + '\n')
    out_file.flush()
    for dat in data_sets:
        for knn in ks:
            for d in dim:
                for n in samp_sizes:
                    data, info, dataTitle = dat(n,d)
                    cond_set = list(range(2, 2+d))
                    start = t.time()
                    cmiEst = k.cmi([0], [1], cond_set, knn, data)
                    end = t.time()
                    run_time = end - start
                    out_list = [n, d, knn, dat.__name__, run_time, cmiEst, info, dataTitle]
                    out_str = ','.join(str(e) for e in out_list) + '\n'
                    out_file.write(out_str)
                    out_file.flush()
    out_file.close()
    pass

def simCont(filename, dim = [1,5,9], ks = [2,6], samp_sizes = range(100, 1001, 100)):
    data_sets = [contDep]
    col_names = ['sampSize', 'dim', 'knn', 'data', 'runTime', 'cmiEst', 'truth', 'dataTitle']
    fname = filename + '.csv'
    out_file = open(fname, 'w+')
    out_file.write(','.join(col_names) + '\n')
    out_file.flush()
    for dat in data_sets:
        for knn in ks:
            for d in dim:
                for n in samp_sizes:
                    data, info, dataTitle = dat(n,d)
                    cond_set = list(range(2, 2+d))
                    start = t.time()
                    cmiEst = k.cmi([0], [1], cond_set, knn, data, minzero = 0)
                    end = t.time()
                    run_time = end - start
                    out_list = [n, d, knn, dat.__name__, run_time, cmiEst, info, dataTitle]
                    out_str = ','.join(str(e) for e in out_list) + '\n'
                    out_file.write(out_str)
                    out_file.flush()
    out_file.close()
    pass

    
if __name__ == '__main__':

    np.random.seed(9345)
    filename = 'standSim'
    sim(filename)
    os.system('python3 visualize.py ' + filename)
    
