## Coreset Util
## Karan Vombatkere, Dec 2021

import pandas as pd
import numpy as np
import os

class Coreset_Util:
    """
    Class for Coreset Utility functions to import, preprocess data. Only float64, int64 data is currently supported
    Parameters
    ----------
        datasetName : The data set name to generate coreset from.
                        Note that the dataset csv must be saved in the data folder
        columnNames :  List of column names to use from dataset, optional
    ----------
    Coreset_Util.X_array contains the formatted ndarray 
    """

    #Initialize with dataset name
    def __init__(self, datasetName, columnNames=None, col_sep=None):
        self.dataframe = self.importData(datasetName, col_sep)
        if columnNames != None:
            self.subsetColumns(columnNames)
        
        self.X_array = self.convertToArray()
    
    #Use Pandas to import dataset - note that data must be stored in data folder and script run from test folder
    def importData(self, fileName, col_sep):
        filePath = os.getcwd()[:-4] + 'data/' + fileName
        importedData = pd.read_csv(filePath, sep = col_sep)

        print('Imported dataset:', fileName)
        return importedData

    #Specify subset of columns to use
    def subsetColumns(self, columnList):
        print('Subsetting Dataset to columns:', columnList)
        self.dataframe = self.dataframe[columnList]
        return None

    #Convert data to Numpy array
    def convertToArray(self):
        #Select only numerical columns
        numerical_dataframe = self.dataframe.select_dtypes(include=['float64', 'int64'])
        x_vals = numerical_dataframe.to_numpy()
        return x_vals

    def generate_Random2DArr(self, size):
        random_2D_vector = []
        for i in range(size):
            x_val, y_val = np.random.randint(-100,101), np.random.randint(-100,101)
            random_2D_vector.append([x_val, y_val])

        return random_2D_vector
        
