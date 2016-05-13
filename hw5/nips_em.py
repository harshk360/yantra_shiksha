import math, random, copy
import numpy as np
import sys
from scipy import misc
from scipy.spatial import distance
from scipy.cluster.vq import vq, kmeans, whiten
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt


def expectation_maximization(x, nbclusters, nbiter=30, epsilon=0.0001):

    def closest_cluster(words_for_doc, mus, nbclusters):
        distances = np.zeros((nbclusters,))
        for j in xrange(nbclusters):
            distances[j] = distance.euclidean(words_for_doc, mus[j])
        return np.argmin(distances)

    def calculate_q(x, nbclusters, mus, pies, w):
        sigma = 0
        for j in xrange(nbclusters):
            inner_prod = x*np.log(mus[j])
            sum = inner_prod + math.log(pies[j])
            sigma += sum*w[:,j,np.newaxis]
        return np.sum(sigma)


    #E step, compute w_i,j
    #vector of pies - pies - [j]
    #vector of mus - mus - [j,k]
    #x - [i,k]
    #w should be [i,j]
    #logA - [i,j]
    #logA_max - [i,1]
    #logY - log(w_ij)
    def e_step(x, nbclusters, mus, pies):
        logA = np.zeros((x.shape[0], nbclusters))
        for j in xrange(nbclusters):
            sigma = x*np.log(mus[j])
            logA[:,j] = np.log(pies[j]) + np.sum(sigma, axis=1)
        logA_max = np.zeros((x.shape[0],))
        logA.max(axis=1, out=logA_max)
        sum=0
        for j in xrange(nbclusters):
            sum += np.exp(logA[:,j] - logA_max)
        term3 = np.log(sum)
        logY = np.zeros((x.shape[0], nbclusters))
        for j in xrange(nbclusters):
            logY[:,j] = logA[:,j] - logA_max - term3
        y = np.exp(logY)
        w = y
        return w


    def m_step(x, w, nbclusters):
        new_mus = np.zeros((nbclusters, x.shape[1]))
        new_pies = np.zeros((nbclusters))
        for j in xrange(nbclusters):
            den = np.sum(np.sum(x, axis=1)*w[:,j])
            num = np.sum(x*w[:,j,np.newaxis], axis=0)
            new_mus[j] = num/den
            new_pies[j] = np.sum(w[:,j])/1500
        new_new_mus = np.zeros((nbclusters, x.shape[1]))
        for j in xrange(nbclusters):
            new_new_mus[j] = (new_mus[j]+.0001)/(np.sum(new_mus[j])+new_mus.shape[1]/10000)
        return new_new_mus, new_pies


    #USE K-MEANS TO GET INITIAL CENTERS
    centroids, distortion = kmeans(x,k_or_guess=nbclusters, iter=5)
    #normalizes ps ie mus
    mus = np.zeros((nbclusters, centroids.shape[1]))
    for j in xrange(nbclusters):
        mus[j] = (centroids[j]+.0001)/(np.sum(centroids[j])+centroids.shape[1]/10000)

    pies = np.full((nbclusters), 1.0/nbclusters)
    iter = 0
    difference = 10000
    old_q = 0
    q = 0
    while iter < 30 and difference > epsilon:
        iter += 1
        print "running iteration " + str(iter)
        w = e_step(x, nbclusters, mus, pies)
        mus, pies = m_step(x, w, nbclusters)
        old_q = q
        q = calculate_q(x, nbclusters, mus, pies, w)
        difference = abs(q-old_q)/abs(q)
        print "Difference in quality is " + str(difference)

    result = {}
    result['clusters'] = {}
    result['params'] = {}
    for i in xrange(nbclusters):
        result['params'][i] = {}
        result['params'][i]['pi'] = pies[i]
        result['params'][i]['mu'] = mus[i]
    for index, words_for_doc in enumerate(x):
        cluster = closest_cluster(words_for_doc, mus, nbclusters)
        if cluster not in result['clusters']:
            result['clusters'][cluster]=[]
        result['clusters'][cluster].append(index)

    #find top 10 words for each cluster
    print ""
    print "top 10 words for each cluster"
    data = [line.strip() for line in open("vocab.nips.txt", 'r')]
    for i in xrange(nbclusters):
        top10 = result['params'][i]['mu'].argsort()[-10:][::-1]
        top10_words = [data[index] for index in top10]
        print top10_words

    return result

#-----------------------------------------------------------------------
num_clusters = 30
data = np.loadtxt('docword.nips.txt', skiprows=3)
#k is total number of words
k = np.max(data[:,1])
i = 1500
#x - [i,k]
x = np.zeros((i,k))
for observation in data:
    x[observation[0]-1][observation[1]-1] = observation[2]

# for vector in x:
#     for word in vector:
#         word += 1
print x
sdsdsd
result = expectation_maximization(x, num_clusters)
new_pies = np.zeros(num_clusters)
for i in xrange(num_clusters):
    new_pies[i]=result['params'][i]['pi']

x_s = np.zeros(30)

for i in xrange(30):
    x_s[i] =i
fig, ax= plt.subplots()
ax.set_xlim([0,30])
ax.set_ylim([0,np.max(new_pies)+.01])
plt.plot(new_pies)
plt.scatter(x_s,new_pies,s=100)
plt.ylabel('Probablities')
#plt.yticks(.01)
plt.xlabel('Topics')
plt.show()
