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

plt.rcParams.update({'font.size': 16})

#################################
## USER EDITABLE CODE FOLLOWS  ##
## INTERSECTIONS WILL BE FOUND ##
## BETWEEN PLOTTED CONTOURS    ##
#################################

picklefile = '../../Open_POPCON/my_popcon_scans/ARCHpcbase.p'

c1name = 'Paux' # first contour name
c2name = 'Pfus' # second contour name

c1lev = 0 # first contour value
c2lev = 2000 # second contour value

with open(picklefile, 'rb') as fp:
    data = pickle.load(fp)

xx = data['xx']
yy = np.tile(data['n20'], (len(data['xx'][0]),1)).transpose() # data['yy']
yG = yy[:,0][np.abs(data['yy'][:,0]-1).argmin()] # where is the Greenwald limit


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
            data[key] = interpolate.interp2d(xx[0], yy[:,0], data[key], kind='cubic')(xnew, ynew)
    xx, yy = np.meshgrid(xnew, ynew)

# plot the contours
plt.figure(figsize=(9,6))
ax = plt.subplot(111)

contour1 = plt.contour(xx, yy, np.transpose(data[c1name]), colors='r', levels=[c1lev], alpha=0)
contour2 = plt.contour(xx, yy, np.transpose(data[c2name]), colors='k', levels=[c2lev], alpha=1, linewidths=5, zorder=9999)

# only plot the stable solutions for paux=0
points1x, points1y = contour_points(contour1).transpose()
c1itop = np.argmax(np.gradient(points1x)) + 1
c1imin = np.argmin(points1y[c1itop:]) + c1itop
plt.plot(points1x[c1itop:c1imin],points1y[c1itop:c1imin],c='r',lw=5, zorder=9999)

#plt.text(0.98,0.13,c1name,color='r',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')
#plt.text(0.98,0.08,c2name,color='k',transform=ax.transAxes,ha='right',va='bottom',fontweight='bold')


# plot Paux contours
contour1_ = plt.contour(xx, yy, np.transpose(data['Paux']), colors='r', levels=[0,1,3,5,10,30,100])

# plot Pfus contours
contour2_ = plt.contour(xx, yy, np.transpose(data['Pfus']), colors='k', levels=[10,50,200,500,1000,3500])
#contour2_ = plt.contour(xx, yy, np.transpose(data['Pfus']), colors='k', levels=[0,10,20,50,100,200,500,1000,2000,3500])

# plot Q contours
contour3 = plt.contour(xx, yy, np.transpose(data['impfrac']), colors='b', levels=[2e-5, 2e-4, 1e-3, 2e-3, 3e-3])

# plot impurity fraction contours
#contour4 = plt.contour(xx, yy, np.transpose(data['impfrac']*100), colors='b', levels=[0.01,0.02,0.05,0.1,0.2])


# make custom legends
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='k', lw=4),
                Line2D([0], [0], color='b', lw=4)]
ax.legend(custom_lines,[r'$P_{aux}$ (MW)',r'$P_{fus}$ (MW)','$n_{imp}/n_{e}$'],loc=3,framealpha=1)

# find the intersections
try:
    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    intersection_points = intersection(points1, points2, eps)
    intersection_points = cluster(intersection_points, cluster_size)

    # plot the intersections
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], s=600, c='gold', ec='k', marker='*', zorder=10000)

    #plt.text(0.98,0.02,r'Intersection at Ti = {:2.1f}, n/n$_G$ = {:1.2f}'.format(intersection_points[:, 0][0], intersection_points[:, 1][0]), transform=ax.transAxes, ha='right', va='bottom')

    # print the intersections
    #print(intersection_points[:, 0][0], intersection_points[:, 1][0])

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

plt.ylim(np.min(np.min(yy)),np.max(np.max(yy)))
plt.ylabel(r'$\overline{n}$ $(10^{20}$ m$^{-3})$')
plt.xlabel(r'$T_i$ (keV)')

plt.fill_between(auxx[ind:],auxy[ind:],8,color='r',alpha=0.3, zorder=-10)


## Add labels! ##

#plt.text(0.47,0.27,"Cordey pass",transform=ax.transAxes,ha='center',va='center',c='r',rotation=-15,
#            bbox=dict(boxstyle="round",ec='w',fc='w',alpha=0.8))

rbbox = dict(boxstyle="round",ec=(255/255,178/255,178/255),fc=(255/255,178/255,178/255),pad=0.2)
wbbox = dict(boxstyle="round",ec='w',fc='w',alpha=1,pad=0.1)

plt.text(0.55,0.9,"Ignition region",transform=ax.transAxes,ha='center',va='center',c='r',bbox=rbbox, zorder=10000)
plt.text(0.12,0.72,"$n_{G}$ limit",transform=ax.transAxes,ha='center',va='center',c='g',bbox=wbbox)

plt.axhline(y=yG, c='g', lw=2,zorder=1000)

## MANUALLY SELECT CONTOR LABELS ##

#ax.clabel(contour1_, fmt='%1.0f')#, manual=True)
#plt.text(0.52,0.51,"0",transform=ax.transAxes,ha='center',va='center',c='r',bbox=rbbox,rotation=0)
plt.text(0.56,0.44,"1",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=16, zorder=9998)
plt.text(0.35,0.09,"1",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=-9)
plt.text(0.4,0.19,"3",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=-5)
plt.text(0.45,0.37,"3",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=-35)
plt.text(0.3,0.3,"5",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=10)
plt.text(0.64,0.145,"5",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=-30)
plt.text(0.2,0.485,"10",transform=ax.transAxes,ha='center',va='center',c='r',bbox=wbbox,rotation=-12)

#ax.clabel(contour2_, fmt='%1.0f')#, manual=True)
#ax.clabel(contour2, fmt='%1.0f')#, manual=True)
plt.text(0.475,0.031,"10",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-4)
plt.text(0.7,0.075,"50",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-4)
plt.text(0.875,0.15,"100",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-5)
plt.text(0.93,0.265,"500",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-2)
plt.text(0.93,0.4,"1000",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-7)
plt.text(0.88,0.605,"2000",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-12, zorder=10000)
plt.text(0.88,0.815,"3500",transform=ax.transAxes,ha='center',va='center',c='k',bbox=wbbox,rotation=-15)

#ax.clabel(contour3, fmt='%1.0e')#, manual=True)
plt.text(0.21,0.9,"2e-5",transform=ax.transAxes,ha='center',va='center',c='b',bbox=wbbox,rotation=-70)
plt.text(0.315,0.9,"2e-4",transform=ax.transAxes,ha='center',va='center',c='b',bbox=wbbox,rotation=-80)
plt.text(0.7,0.38,"1e-3",transform=ax.transAxes,ha='center',va='center',c='b',bbox=wbbox,rotation=-35)
plt.text(0.83,0.45,"2e-3",transform=ax.transAxes,ha='center',va='center',c='b',bbox=wbbox,rotation=-60)
plt.text(0.945,0.9,"3e-3",transform=ax.transAxes,ha='center',va='center',c='b',bbox=wbbox,rotation=-90)

plt.tight_layout()
#plt.show()
plt.savefig('ARCHpcbase.png')
