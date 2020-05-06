import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_variable(inputDir,sample):
    if not os.path.exists('/home/seohyun/work/heptool/deepAna/keras/var'):
        os.makedirs('/home/seohyun/work/heptool/deepAna/keras/var')

    signalHistSet = []
    bkgHistSet = []

    df = pd.read_hdf(inputDir+'/'+sample)
    sigEvent = df.query('signal == 1')
    bkgEvent = df.query('signal == 0')

    variables = list(df)[2:]

    for index, item in enumerate(variables):
        #plt.subplot(nvar/2, nvar/2, index+1)
        nbins = 30
        low = min(sigEvent[item])
        high = max(sigEvent[item])
        edge=(low,high)
        #plt.title(item)
        plt.hist(sigEvent[item], color='b', alpha=0.5, density=True, bins=nbins, range=edge, histtype='stepfilled', label='signal')
        plt.hist(bkgEvent[item], color='r', alpha=0.5, density=True, bins=nbins, range=edge, histtype='stepfilled', label='background')
        plt.xlabel(item)
        plt.legend(loc='upper right')
        plt.savefig('var/'+item+'.pdf')
        plt.close()
        #plt.show()

