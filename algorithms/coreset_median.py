## Median estimation using Coresets
## Karan Vombatkere, Dec 2021

import numpy as np

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
        self.x_array = sorted([x[0] for x in x_arr])
        print(len(self.x_array))
        self.epsilon = epsilon
        

    #Partition and create coresets    
    def partitionSubsequences(self):
        numPartitions = int(1/self.epsilon) + 1
        partitionSize = int(len(self.x_array)/numPartitions)

        coreset_Arr = []
        for i in range(0, len(self.x_array), partitionSize):
            coreset_Arr.append(self.x_array[i])

        return coreset_Arr

    #Compute median value
    def compute_median(self):
        coreset_array = self.partitionSubsequences()
        medianVal = coreset_array[int(len(coreset_array)/2)]
        
        return medianVal


