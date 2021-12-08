"""
Class implementing the streaming coresets algorithm from
"Coresets for k-Means and k-Median Clustering and their Applications"
by Har-Peled and Mazumdar, 2003. https://arxiv.org/pdf/1810.12826.pdf

TODO:
fix side length -> depends on whether we are doing kmeans or kmedians. also
will require eps value, to compute R.

viz -> on simulated data of 4 gaussian clusters
run weighted k-means
run weighted k-median
compute costs
benchmark
"""

import numpy as np
import math
import random
import warnings

class Coreset_Streaming:
    def __init__(self, max_size):
        """ Intitializes the Coreset_Streaming class.
            Args:
                max_size (Int): maximum size we allow  the coreset to get before
                                doubling the resolution
        """
        self.resolution = 1         # level of resolution
        self.side_length = 2        # side_length of d-dimensional grid
        self.max_size = max_size    # maximum size of coreset before resolution increase
        self.coreset = []           # arr to store coreset
        self.grid_points = dict()   # keys are grid_points, values are arrays of points

    def add_point(self, point):
        """ Adds point `point` to the coreset, with a default weight of 1.
        """
        weight = 1
        self.coreset.append((point, weight))

    def grid_location(self, point):
        """ Returns the location in the grid of the argument `point`.
            We identify grid locations by the smallest value of each dimension
            of that box in the grid.
        """
        loc = []
        for dim in point:
            loc.append(dim - dim % self.side_length)
        return loc

    def snap_points_to_grid(self):
        """ Snaps each point in the coreset to a box in the grid, and populates
            self.grid_points with the point.
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
        """ Goes to each populated grid point and chooses a representative
            point from that box in the grid. This representative gets the
            cumulative weight of all the points in this box. The rest of the
            points are discarded.
            Assumes points have already been snapped to the grid.
        """
        self.coreset = []
        for (loc, points) in self.grid_points.items():
            (representative, _) = random.choice(self.grid_points[loc])
            repr_weight = 0
            for (point, weight) in self.grid_points[loc]:
                repr_weight += weight
            self.coreset.append((representative, repr_weight))

    def build_coreset(self):
        """ Builds a coreset.
            This function can be run multiple times without messing up the coreset.
        """
        self.snap_points_to_grid()
        self.build_coreset_from_grid()
        if len(self.coreset) > self.max_size:
            self.double_resolution()

    def double_resolution(self):
        """ Doubles the resolution of the grid i.e. if the box was 2x2 previously
            (in 2D) then it becomes 4x4.
        """
        print("Doubling resolution rank from {} to {}".format(self.resolution,
                                                              self.resolution+1))
        self.resolution += 1
        self.side_length *= 2
        self.build_coreset()

    def can_union(self, cs):
        """ Returns True if the other Coreset can be "union"-ed with
            self.
        """
        if cs.resolution == self.resolution:
            return True
        else:
            warnings.warn("Current Coreset resolution is {}, other's resolution"
                          "is {}".format(self.resolution, cs.resolution))
            return False

    def union(self, cs):
        """ Adds the other coreset `cs` to current coreset.
        """
        # check if resolution is the same.
        if self.resolution != cs.resolution:
            raise Exception("Current Coreset resolution is {}, other's resolution"
                            "is {}".format(self.resolution, cs.resolution))

        # add points to self's coreset
        for (point, weight) in cs.coreset:
            self.coreset.append((point, weight))

        self.build_coreset()
