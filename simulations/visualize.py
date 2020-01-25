import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf
import os
import sys

np.random.seed(1234)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown', 'yellow']
linetypes = ['solid', 'dashed', 'dashdot', 'dotted']

def datak(path):

    testDat = pd.read_csv(path + ".csv")
    
    dims = list(set(testDat['dim']))
    ks = list(set(testDat['knn']))
    dats = list(set(testDat['data']))
    
    grf = matplotlib.backends.backend_pdf.PdfPages(path + ".pdf")
    for dat in dats:
        for k in ks:
            fig = plt.figure()
            for dimCount in range(len(dims)):
                curDat = testDat.loc[(testDat['dim'] == dims[dimCount]) & (testDat['data'] == dat) &
                                     (testDat['knn'] == k)]
                plt.plot('sampSize', 'cmiEst', data=curDat, marker='o', markersize=4,
                         color= colors[dimCount], linewidth=1, label = dims[dimCount])
            trueInfo = list(curDat['truth'])[0]
            plt.axhline(y= trueInfo, color='black', linestyle='--')
            title = list(curDat['dataTitle'])[0] + ', k = '
            plt.title(title + str(k))
            plt.legend(title = 'Dimension')
            plt.xlabel("Sample Size")
            plt.ylabel("Estimated Conditional Mutual Information")
            grf.savefig(fig)
    grf.close()
    pass

def together(path):

    testDat = pd.read_csv(path + ".csv")
    dims = list(set(testDat['dim']))
    ks = list(set(testDat['knn']))
    dats = list(set(testDat['data']))
    
    grf = matplotlib.backends.backend_pdf.PdfPages(path + ".pdf")
    for dat in dats:
        fig = plt.figure()
        for kCount in range(len(ks)):
            for dimCount in range(len(dims)):
                curDat = testDat.loc[(testDat['dim'] == dims[dimCount]) & (testDat['data'] == dat) &
                                     (testDat['knn'] == ks[kCount])]
                plt.plot('sampSize', 'cmiEst', data=curDat, marker='o', markersize=4,
                         color= colors[dimCount], linestyle = linetypes[kCount], linewidth=1,
                         label = 'd = ' + str(dims[dimCount]) + ', k = ' + str(ks[kCount]))
        trueInfo = list(curDat['truth'])[0]
        plt.axhline(y= trueInfo, color='black', linestyle='--')
        title = list(curDat['dataTitle'])[0]
        plt.title(title)
        plt.legend()
        plt.xlabel("Sample Size")
        plt.ylabel("Estimated Conditional Mutual Information")
        grf.savefig(fig)
    grf.close()
    pass

def togetherPanel(path):
    # not working
    testDat = pd.read_csv(path + ".csv")
    
    dims = list(set(testDat['dim']))
    ks = list(set(testDat['knn']))
    dats = list(set(testDat['data']))
    
    grf = matplotlib.backends.backend_pdf.PdfPages(path + ".pdf")
    fig = plt.figure()
    counter = 1
    for dat in dats:
        sfig = plt.subplot(2,2,counter)
        counter =+ 1
        title = dat
        for kCount in range(len(ks)):
            for dimCount in range(len(dims)):
                curDat = testDat.loc[(testDat['dim'] == dims[dimCount]) & (testDat['data'] == dat) &
                                     (testDat['knn'] == ks[kCount])]
                sfig.plot('sampSize', 'cmiEst', data=curDat, marker='o', markersize=4,
                         color= colors[dimCount], linestyle = linetypes[kCount], linewidth=1,
                         label = 'd = ' + str(dims[dimCount]) + ', k = ' + str(ks[kCount]))
        trueInfo = list(curDat['truth'])[0]
        sfig.axhline(y= trueInfo, color='black', linestyle='--')
        sfig.title(title)
    plt.xlabel("Sample Size")
    plt.ylabel("Estimated Conditional Mutual Information")
    plt.legend()
    grf.savefig(fig)
    grf.close()
    pass


if __name__ == '__main__':

    path = sys.argv[1]
    together(path)
    os.system('open ' + path + '.pdf')
