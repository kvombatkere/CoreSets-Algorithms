## Coreset Implementation - Minimum Enclosing Ball
## Karan Vombatkere, Dec 2021

import numpy as np
import matplotlib.pyplot as plt
import time

class Coreset_MinimumEnclosingBall:
    """
    Class to compute Minimum Enclosing Ball Coreset
    Parameters
    ----------
        x_arr : numpy array data from Coreset_Util - must be R^2
        epsilon : epsilon value
    ----------
    Reference: Chapter 12, Data Stream Algorithms Lecture Notes, Amit Chakrabarti
    """
    
    #Initialize with parameters
    def __init__(self, x_arr, epsilon, plot_flag=False):
        if isinstance(x_arr, np.ndarray):
            self.x_array = x_arr
        else:
            self.x_array = np.array(x_arr)

        self.epsilon = epsilon

        self.meb_vector = []
        self.plotFlag = plot_flag


    #Compute theta grid for data
    def compute_thetaGrid(self):
        thetaVal = np.sqrt(self.epsilon)
        numAngles = int((2*np.pi)/(thetaVal)) + 1
        
        self.thetaAngles = []
        self.angleSize = (2*np.pi/numAngles)

        for i in range(numAngles):
            angle_i = self.angleSize*i
            y_val = np.tan(angle_i)
            x_val = 1

            if (np.pi/2) <= angle_i <= (3*np.pi/2):
                x_val = -x_val
                y_val = -y_val

            self.thetaAngles.append([x_val, y_val])

        return


    #Function to compute angle between two vetors
    def vector_angle(self, u, v):
        dotProduct = np.dot(u,v)
        l2_normProduct = np.linalg.norm(u)*np.linalg.norm(v)
        angle = np.arccos(dotProduct/l2_normProduct)
        
        return angle


    #Function to compute argmin angle between a vector and x_array
    def compute_mebAngle_vector(self, vec):
        maxProduct = 0
        mebVec = []
        for u in self.x_array:
            dotProduct = np.dot(u, vec)
            vec_angle = self.vector_angle(u, vec)

            #Consider points only along direction of vec
            if vec_angle <= self.angleSize:
                if dotProduct >= maxProduct:
                    mebVec = u
                    maxProduct = dotProduct

        return mebVec


    #Function to plot MEB and x_array
    def plot2D_meb(self):
        x_plt = [vec[0] for vec in self.x_array]
        y_plt = [vec[1] for vec in self.x_array]

        plt.scatter(x_plt, y_plt, s= 0.05, label = 'Points')

        meb_x = [vec[0] for vec in self.meb_vector if isinstance(vec, np.ndarray)]
        meb_y = [vec[1] for vec in self.meb_vector if isinstance(vec, np.ndarray)]

        plt.scatter(meb_x, meb_y, color='red', marker = '+', label = 'MEB')

        title_text = 'Minimum enclosing ball, epsilon = {}'.format(self.epsilon)
        plt.title(title_text, fontsize=11)
        plt.ylabel('y')
        plt.xlabel('x')
        # plt.xlim([min(x_plt) - abs(0.1*min(x_plt)), max(x_plt) + 0.1*max(x_plt)])
        # plt.ylim([min(y_plt) - abs(0.1*min(y_plt)), max(y_plt) + 0.1*max(y_plt)])

        plt.legend(loc='lower right', fontsize=9)
        plt.rcParams["figure.figsize"] = (8,8)
        plt.show()

        return None


    #Compute Minimum Enclosing ball
    def compute_minimumEnclosingBall(self):
        startTime = time.perf_counter()

        #Compute theta grid
        self.compute_thetaGrid()

        for vec_i in self.thetaAngles:
            mebVec_i = self.compute_mebAngle_vector(vec_i)
            self.meb_vector.append(mebVec_i)

        print("Computed Minimum Enclosing Ball of size = {}".format(len(self.meb_vector)))

        if self.plotFlag:
            self.plot2D_meb()

        runTime = time.perf_counter() - startTime
        print("Minimum Enclosing Ball computation time = {:.1f} seconds".format(runTime))

        return 
        