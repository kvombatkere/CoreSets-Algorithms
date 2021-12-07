"""
Class implementing something something TODO

Bhushan Suwal
Dec 2021

Procedure:
keep adding points
snap()
build_coreset()

TODO:
implement coresets
"""

import numpy as np
import math
import random

class Coreset_Streaming:
    def __init__(self):
        self.resolution = 1
        self.side_length = 2
        # keys are points, values are weights
        self.coreset = dict()
        self.point_bucket = [] # stores points before building point_bucket
        self.grid_points = dict() # keys are grid_points, values are arrays of points

    def add_point_to_bucket(self, point):
        """
        """
        self.point_bucket.append(point)

    def grid_location(self, point):
        """ We identify grid locations by the smallest value of each dimension
            of that box in the grid.
        """
        loc = []
        for dim in point:
            loc.append(dim - dim % side_length)
        return np.array(loc)

    def snap(self):
        """
        """
        for point in self.point_bucket:
            loc = self.grid_location(point)
            if loc in self.grid_points.keys():
                self.grid_points[loc].append(point)
            else:
                self.grid_points[loc] = [point]

    def build_coreset(self):
        """ Assumes points have already been "snapped".
        """
        for (loc, points) in self.grid_points:
            representative = random.choice(self.grid_points[loc])
            weight = len(self.grid_points[loc])
            self.coreset[representative] = weight
        self.point_bucket = []

    def double_resolution(self):
        """
        """
        resolution *= 2
        side_length *= 2
        self.snap()
