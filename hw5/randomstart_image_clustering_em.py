import math, random, copy
import numpy as np
import sys
from scipy import misc
from scipy.spatial import distance
from scipy.cluster.vq import vq, kmeans, whiten
from numpy.linalg import inv


def expectation_maximization(x, nbclusters, nbiter, epsilon=0.0001):

    def closest_cluster(pixel, mus, nbclusters):
        pixel_multiple = np.zeros((nbclusters,3))
        for i in xrange(nbclusters):
            pixel_multiple[i] = pixel
        distances = distance.cdist(pixel_multiple, mus, 'euclidean')
        return np.argmin(distances)

    def find_dmin(x, mus):
        distances = distance.cdist(x, mus, 'euclidean')
        dmin = np.amin(distances, axis=1)
        return dmin

    def calculate_q(x, nbclusters, mus, pies, w):
        sigma = 0
        for j in xrange(nbclusters):
            exped = pies[j]*np.exp(-1*((((x-mus[j])**2).sum(1)))/2.0)
            dot = -1*(exped)/2.0
            sigma += (dot + pies[j])*w[:,j]
        return np.sum(sigma)


    #E step, compute w_i,j
    #vector of pies - pies - [10]
    #vector of mus - mus - [10,3]
    #x - [240000,3]
    #w should be [240000,10]
    def e_step(x, nbclusters, mus, pies):
        denominator = np.zeros((x.shape[0]))
        w = np.zeros((x.shape[0], nbclusters))
        dmin = find_dmin(x, mus)
        dmin_sqd = np.square(dmin)
        for k in xrange(nbclusters):
            exped = pies[k]*np.exp(-1*((((x-mus[k])**2).sum(1))-dmin_sqd)/2.0)
            denominator += exped
        for k in xrange(nbclusters):
            exped = pies[k]*np.exp(-1*((((x-mus[k])**2).sum(1))-dmin_sqd)/2.0)
            w[:,k] = exped/denominator
        return w


    def m_step(x, w, nbclusters):
        new_mus = np.zeros((nbclusters, 3))
        new_pies = np.zeros((nbclusters))
        three_w = np.zeros((w.shape[0],w.shape[1],3))
        three_w[:,:,0] = w
        three_w[:,:,1] = w
        three_w[:,:,2] = w
        for j in xrange(nbclusters):
            den = np.sum(w[:,j])
            num = np.sum(x*three_w[:,j,:], axis=0)
            new_mus[j] = num/den
            new_pies[j] = den/(x.shape[0])
        return new_mus, new_pies


    whitened = x
    # #USE K-MEANS TO GET INITIAL CENTERS
    # centroids, distortion = kmeans(whitened,k_or_guess=nbclusters, iter=5)
    # mus = centroids

    #RANDOMLY GENERATING START CENTERS
    mus = np.zeros((nbclusters,3))
    for i in xrange(nbclusters):
       mus[i] = whitened[np.random.random_integers(whitened.shape[0]-1)]

    print ""
    print "Random start points are "
    for i in xrange(nbclusters):
        print mus[i]

    pies = np.full((nbclusters), 1.0/nbclusters)
    iter = 0
    difference = 10000
    old_q = 0
    q = 0
    while iter < 30 and difference > epsilon:
        iter += 1
        print "running iteration " + str(iter)
        w = e_step(whitened, nbclusters, mus, pies)
        mus, pies = m_step(whitened, w, nbclusters)
        old_q = q
        q = calculate_q(whitened, nbclusters, mus, pies, w)
        difference = abs(q-old_q)/abs(q)
        print "Difference in quality is " + str(difference)

    result = {}
    result['clusters'] = {}
    result['params'] = {}
    for i in xrange(nbclusters):
        result['params'][i] = {}
        result['params'][i]['pi'] = pies[i]
        result['params'][i]['mu'] = mus[i]
    for index, pixel in enumerate(whitened):
        cluster = closest_cluster(pixel, mus, nbclusters)
        if cluster not in result['clusters']:
            result['clusters'][cluster]=[]
        result['clusters'][cluster].append(index)

    return result

#-----------------------------------------------------------------------
num_clusters = 20
for round in xrange(5):
    arr = misc.face()
    arr = misc.imread('images/polarlights.jpg')

    arr_new = []
    numpy_array = np.empty([len(arr)*len(arr[0]), 3])
    for row in arr:
        for pixel in row:
            arr_new.append(pixel)
    for index, item in enumerate(arr_new):
        numpy_array[index] = item


    result = expectation_maximization(numpy_array, nbclusters = num_clusters, nbiter=1)


    write_array = np.empty([len(arr), len(arr[0]), 3])
    for i in xrange(num_clusters):
        for item in result['clusters'][i]:
            row = item//len(arr[0]);
            col = item - row*len(arr[0]);
            write_array[row][col] = result['params'][i]['mu']
    name = 'clustered_polarlights_20_random_' + str(round) + '.jpg'
    misc.imsave(name, write_array)


