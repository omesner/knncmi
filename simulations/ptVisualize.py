import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf

ptAn = pd.read_csv('pointAnalysis.csv')
dats = set(ptAn['data'])
dims = set(ptAn['dim'])
ks = set(ptAn['knn'])

grf = matplotlib.backends.backend_pdf.PdfPages("pointHists.pdf")
for dat in dats:
    title = dat + ', k = '
    for dim in dims:
        for k in ks:
            data = ptAn.loc[(ptAn['data'] == dat) & (ptAn['dim'] == dim) & (ptAn['knn'] == k)]
            est = list(data['cmiEst'])[0]
            trueInfo = list(data['trueInfo'])[0]
            fig = plt.figure()
            plt.hist(data['ptEst'], bins = 1000)
            plt.axvline(x = trueInfo, color='black', linestyle='--')
            plt.text(trueInfo, 0, 'Truth', color='black',rotation=90)
            plt.axvline(x = list(data['cmiEst'])[0], color = 'red', linestyle = '--', label = 'Estimate')
            plt.text(est, 0, 'Estimate', color='red',rotation=90)
            plt.title(title + str(int(k)) + ', Dim = ' + str(int(dim)))
            grf.savefig(fig)

grf.close()


data = ptAn.loc[(ptAn['data'] == dat) & (ptAn['dim'] == dim) & (ptAn['knn'] == k)]
