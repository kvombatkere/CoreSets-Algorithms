"""
Class implementing something something TODO

Bhushan Suwal
Dec 2021

TODO:
implement how set union would work
test union
run it on a stream
figure initial resolution
numpy-ify everything
clean everything
"""

import numpy as np
import math
import random

class Coreset_Streaming:
    def __init__(self):
        self.resolution = 1
        self.side_length = 2
        self.coreset = [] # corresponding index has weight in weights
        self.grid_points = dict() # keys are grid_points, values are arrays of points

    def add_point(self, point):
        """
        """
        weight = 1
        self.coreset.append((point, weight))

    def grid_location(self, point):
        """ We identify grid locations by the smallest value of each dimension
            of that box in the grid.
        """
        loc = []
        for dim in point:
            loc.append(dim - dim % self.side_length)
        return loc

    def snap_points_to_grid(self):
        """
        """
        self.grid_points = dict()
        for (point, weight) in self.coreset:
            # hack: lists can't be dict keys, so a hack is to turn them into strings
            loc = repr(self.grid_location(point))
            if loc in self.grid_points.keys():
                self.grid_points[loc].append((point, weight))
            else:
                self.grid_points[loc] = [(point, weight)]

    def build_coreset_from_grid(self):
        """ Assumes points have already been "snapped".
        """
        self.coreset = []
        for (loc, points) in self.grid_points.items():
            (representative, _) = random.choice(self.grid_points[loc])
            repr_weight = 0
            for (point, weight) in self.grid_points[loc]:
                repr_weight += weight
            self.coreset.append((representative, repr_weight))

    def build_coreset(self):
        """
        """
        self.snap_points_to_grid()
        self.build_coreset_from_grid()

    def double_resolution(self):
        """
        """
        self.resolution += 1
        self.side_length *= 2
        self.build_coreset()
