## Median estimation using Coresets
## Karan Vombatkere, Dec 2021

import numpy as np
import time

class Coreset_Median:
    """
    Class to compute median using Coresets
    Parameters
    ----------
        x_arr : numpy array data from Coreset_Util - computes on first dimension
        epsilon : epsilon value for Coreset
    """

    #Initialize with data and epsilon value
    def __init__(self, x_arr, epsilon):
        self.x_array = sorted(x_arr)
        #print('Length of input = ', len(self.x_array))
        self.epsilon = epsilon

        self.numPartitions = 0
        

    #Partition and create coresets    
    def partitionSubsequences(self):
        self.numPartitions = int(1/self.epsilon) + 1
        partitionSize = int(len(self.x_array)/self.numPartitions)
        print(partitionSize)

        coreset_Arr = []
        for i in range(0, len(self.x_array), partitionSize):
            coreset_Arr.append(self.x_array[i])

        print("Total elements checked = {}".format(self.numPartitions))

        return coreset_Arr


    #Compute median value
    def compute_median(self):
        #Compute approx median
        startTime = time.perf_counter()

        coreset_array = self.partitionSubsequences()
        self.approxMedianVal = coreset_array[int(len(coreset_array)/2)]

        print("Median value approximation = {}".format(self.approxMedianVal))

        self.runTime = time.perf_counter() - startTime
        print("Median approximation computation time = {:.5f} seconds".format(self.runTime))

        #Compute true median
        np_startTime = time.perf_counter()

        self.trueMedian = np.median(self.x_array)
        print("True Median = {}".format(self.trueMedian))

        self.np_runTime = time.perf_counter() - np_startTime
        print("Numpy Median computation time = {:.5f} seconds".format(self.np_runTime))
        
        return 
