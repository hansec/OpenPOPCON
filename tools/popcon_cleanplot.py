# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:51:21 2020

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

## PLOT PARAMS#

plt.rcParams.update({'font.size': 14})

#################################
## USER EDITABLE CODE FOLLOWS  ##
## INTERSECTIONS WILL BE FOUND ##
## BETWEEN PLOTTED CONTOURS    ##
#################################

picklefile = 'my_popcon_scans/ARCHpcbase.p'

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
plt.figure(figsize=(9,6))

ax = plt.subplot(111)

contour1 = plt.contour(xx, yy, np.transpose(data[c1name]), colors='r', levels=[c1lev], alpha=0)
contour2 = plt.contour(xx, yy, np.transpose(data[c2name]), colors='k', levels=[c2lev], alpha=0)

#plt.text(0.98,0.13,c1name,color='r',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')
#plt.text(0.98,0.08,c2name,color='k',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')

# contour labels
# ax.clabel(contour1)
# ax.clabel(contour2)

# plot Paux contours
contour1_ = plt.contour(xx, yy, np.transpose(data['Paux']), colors='r', levels=[0,1,5,10,50,100])

# plot Pfus contours
contour2_ = plt.contour(xx, yy, np.transpose(data['Pfus']), colors='k', levels=[0,50,100,200,500,1000,2000,3000])

# plot Q contours
contour3 = plt.contour(xx, yy, np.transpose(data['Q']), colors='lime', levels=[1,5,10])

# plot impurity fraction contours
#contour4 = plt.contour(xx, yy, np.transpose(data['impfrac']*100), colors='b', levels=[0.01,0.02,0.05,0.1,0.2])


# make custom legends
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='lime', lw=4)]
ax.legend(custom_lines,[r'$P_{aux}$ (MW)',r'$P_{fus}$ (MW)','Q'],loc=3,framealpha=1)

# find the intersections
try:
    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    intersection_points = intersection(points1, points2, eps)
    intersection_points = cluster(intersection_points, cluster_size)

    # plot the intersections
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=400, c='gold', ec='k', marker='*', zorder=100)

    #plt.text(0.98,0.02,r'Intersection at Ti = {:2.1f}, n/n$_G$ = {:1.2f}'.format(intersection_points[:, 0][0], intersection_points[:, 1][0]), transform=ax.transAxes, ha='right', va='bottom')

    # print the intersections
    print(intersection_points[:, 0][0], intersection_points[:, 1][0])

except ValueError:
    plt.text(0.98,0.02,r'NO INTERSECTION FOUND', transform=ax.transAxes, ha='right', va='bottom')

## ASUMING THAT countour1 IS PAux , FILL ABOVE THE TOP CONTOUR ##
auxx = points1[:,0]
auxy = points1[:,1]
# find where aux starts decreasing = top contour
for i,x in enumerate(auxx[1:]):
    if x<auxx[i]:
        ind = i
        break
plt.fill_between(auxx[ind:],auxy[ind:],1,color='r',alpha=0.3)

plottitle = picklefile.split('_')[-1].split('.p')[0].split('-')
import re
def separate_number_chars(s):
    res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', s.strip())
    res_f = [r.strip() for r in res if r is not None and r.strip() != '']
    return res_f
plottitle = [separate_number_chars(txt) for txt in plottitle]
def unit(s):
    if s == 'Ip':
        return 'MA'
    elif s == 'R':
        return 'm'
    else:
        return ''
plottitle = ',  '.join([txt[0]+'='+txt[1]+unit(txt[0]) for txt in plottitle])
plt.title(plottitle)

plt.ylabel(r'$\bar{n}/n_{GW}$')
plt.xlabel(r'$T_i$ (keV)')

## MANUALLY SELECT CONTOR LABELS ##
ax.clabel(contour1_, fmt='%1.0f', manual=True)
ax.clabel(contour2_, fmt='%1.0f', manual=True)
ax.clabel(contour3, fmt='%1.0f', manual=True)


plt.show()
