# -*- coding: utf-8 -*-
"""
Created on Tue Oct 08 23:26:23 2020

OpenPOPCON
Based on algorithms in
<https://stackoverflow.com/questions/17416268/how-to-find-all-the-intersection-points-between-two-contour-set-in-an-efficient>,
this script can be used to find the intersection points between various contours
in a OpenPOPCON dataset.

Denpendencies:
    -pickle
    -collections
    -numpy
    -scipy
    -matplotlib

@author: nelsonand
"""

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
#import os

### FUNCTION DEFINITIONS ###

def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points

def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])

def contour_points(contour, steps=1):
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)

#################################
## USER EDITABLE CODE FOLLOWS  ##
## INTERSECTIONS WILL BE FOUND ##
## BETWEEN PLOTTED CONTOURS    ##
#################################

#os.chdir('../../')

picklefile = 'my_popcon_scans/ARCHpcbase.p'
plottitle = picklefile[-16:-2]

c1name = 'Paux' # first contour name
c2name = 'Pfus' # second contour name

c1lev = 0 # first contour value
c2lev = 2010 # second contour value

with open(picklefile, 'rb') as fp:
    data = pickle.load(fp)

xx = data['xx']
yy = data['yy']

# every intersection point must be within eps of a point on the other
# contour path
eps = 0.02

# cluster together intersection points so that the original points in each flat
# cluster have a cophenetic_distance < cluster_size
cluster_size = 0.2

# optional - interpolate onto finer grid to increase intersection accuracy
if (True):
    n = 1000
    xnew = np.linspace(np.min(xx), np.max(xx), n)
    ynew = np.linspace(np.min(yy), np.max(yy), n)
    for key in data:
        if key not in ['xx', 'yy']:
            data[key] = interpolate.interp2d(xx[0], yy[:,0], data[key])(xnew, ynew)
    xx, yy = np.meshgrid(xnew, ynew)

# plot the contours
plt.figure()

ax = plt.subplot(111)

contour1 = plt.contour(xx, yy, np.transpose(data[c1name]), colors='r', levels=[c1lev])
contour2 = plt.contour(xx, yy, np.transpose(data[c2name]), colors='k', levels=[c2lev])

plt.title(plottitle)
plt.text(0.98,0.13,c1name,color='r',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')
plt.text(0.98,0.08,c2name,color='k',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')

# contour labels
# ax.clabel(contour1)
# ax.clabel(contour2)

# plot Vloop contours
# contour3 = plt.contour(xx, yy, np.transpose(data['Vloop']), colors='g', levels=[0.01,0.02,0.04,0.08,0.16,0.32])
# ax.clabel(contour3)

# find the intersections
try:
    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    intersection_points = intersection(points1, points2, eps)
    intersection_points = cluster(intersection_points, cluster_size)

    # plot the intersections
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=30)

    plt.text(0.98,0.02,r'Intersection at Ti = {:2.1f}, n/n$_G$ = {:1.2f}'.format(intersection_points[:, 0][0], intersection_points[:, 1][0]), transform=ax.transAxes, ha='right', va='bottom')

    # print the intersections
    print(intersection_points[:, 0][0], intersection_points[:, 1][0])

except ValueError:
    plt.text(0.98,0.02,r'NO INTERSECTION FOUND', transform=ax.transAxes, ha='right', va='bottom')

plt.show()
