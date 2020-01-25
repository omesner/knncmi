import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os

expDat = pd.read_csv('expData.csv')

dats = expDat.data.unique()
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b',
          u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

sampSizes = expDat.sampSize.unique()
dims = expDat.dim.unique()
    
grf = matplotlib.backends.backend_pdf.PdfPages("boxplotsExperiment.pdf")
for dat in dats:
    fig = plt.figure()
    curDat = expDat.loc[(expDat['data'] == dat) & (expDat['knn'] == 7) &
                        (expDat['dim'] == 1)]

    # create list of vectors for violin plots
    fpDat = []
    ravk1Dat = []
    ravk2Dat = []
    propDat = []
    for size in sampSizes:
        plotDat = curDat.loc[curDat['sampSize'] == size]
        fpDat.append(plotDat['FP'].values)
        ravk1Dat.append(plotDat['RAVK1'].values)
        ravk2Dat.append(plotDat['RAVK2'].values)
        propDat.append(plotDat['Proposed'].values)

    trueInfo = list(curDat['trueInfo'])[0]
    plt.axhline(y= trueInfo, color='black', linestyle='--', label = 'True CMI')

    vp1 = plt.violinplot(fpDat, positions = sampSizes, widths = 50)
    vp2 = plt.violinplot(ravk1Dat, positions = sampSizes, widths = 50)
    vp3 = plt.violinplot(ravk2Dat, positions = sampSizes, widths = 50)
    vp4 = plt.violinplot(propDat, positions = sampSizes, widths = 50)

    plt.plot(sampSizes, curDat.groupby(['sampSize']).median()['FP'].values,
             marker = 'x', color = colors[0], label = 'FP')
    plt.plot(sampSizes, curDat.groupby(['sampSize']).median()['RAVK1'].values,
             marker = 'x', color = colors[1], label = 'RAVK1')
    plt.plot(sampSizes, curDat.groupby(['sampSize']).median()['RAVK2'].values,
             marker = 'x', color = colors[2], label = 'RAVK2')
    plt.plot(sampSizes, curDat.groupby(['sampSize']).median()['Proposed'].values,
             marker = 'x', color = colors[3], label = 'Proposed')
    
    title = list(curDat['title'])[0]
    plt.title(title) # this line was commented out for the paper
    plt.legend()
    plt.xlabel("Sample Size")
    plt.ylabel("Estimated I(X;Y|Z)")
    grf.savefig(fig)
grf.close()

os.system('open boxplotsExperiment.pdf')
