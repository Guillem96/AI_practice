from biclusters import bicluster
from math import sqrt
import random
import copy
import sys

def readfile(filename):
    lines=[line for line in file(filename)]
    # First line is the column titles
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        # First column in each row is the rowname
        rownames.append(p[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in p[1:]])
    return rownames,colnames,data


def euclidean(v1, v2):
    total = 0.0
    for i in range(len(v1)):
        total += pow(v1[i] - v2[i],2)

    return sqrt(total)


def pearson(v1,v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of the squares
    sum1Sq = sum([pow(v,2) for v in v1])
    sum2Sq = sum([pow(v,2) for v in v2])
    # Sum of the products
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])
    # Calculate r (Pearson score)
    num = pSum-(sum1 * sum2/len(v1))
    den = sqrt((sum1Sq - pow(sum1,2)/len(v1)) * (sum2Sq-pow(sum2,2) / len(v1)))
    if den == 0: return 0
    return 1.0 - num/den


def hcluster(rows,distance=pearson):
    distances={} # stores the distances for efficiency
    currentclustid=-1 # all except the original items have a negative id

    # Clusters are initially just the rows
    clust = [bicluster(rows[i],id=i) for i in range(len(rows))]

    while len(clust)>1: #stop when there is only one cluster left
        lowestpair = (0,1)
        closest = distance(clust[0].vec,clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)] = distance(clust[i].vec,clust[j].vec)

                d = distances[(clust[i].id,clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i,j)
        # calculate the average of the two clusters
        mergevec = [
            (clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])/2.0 \
            for i in range(len(clust[0].vec))
        ]
        # create the new cluster
        newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
                                right=clust[lowestpair[1]],
                                dist=closest,id=currentclustid)
        # cluster ids that weren t in the original set are negative
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]


def printclust(clust,labels=None,n=0):
    # indent to make a hierarchy layout
    for i in range(n): print ' ',
    if clust.id<0:
        # negative id means that this is branch
        print '-'
    else:
        # positive id means that this is an endpoint
        if labels==None:
            print clust.id
        else:
            print labels[clust.id]
    # now print the right and left branches
    if clust.left!=None:
        printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None:
        printclust(clust.right,labels=labels,n=n+1)


# Without restart polices, used for getting information to improve the result of tha algorithm
# when using restart polices
def _kcluster(rows,distance=pearson,k=4, trials=100, clusters=None):
    if not clusters:
        clusters = []
        for k in range(k):
            clusters.append(generate_random_clusters(rows))

    lastmatches = None
    for t in range(trials):
        bestmatches = [[] for i in range(k)]
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i],row)
                if d < distance(clusters[bestmatch],row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)
        # If the results are the same as last time, done
        if bestmatches == lastmatches: break
        lastmatches = copy.deepcopy(bestmatches)

        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i]=avgs

    centroide_distances = [ 0 for _ in xrange(k)] # square of distances of all items to centroid
    for cluster, best_match in enumerate(bestmatches):
        for item in best_match:
            centroide_distances[cluster] += pow(euclidean(clusters[cluster], rows[item]), 2)
    return sum(centroide_distances), bestmatches, centroide_distances


def generate_random_clusters(rows):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),
            max([row[i] for row in rows])) for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    return [random.random() * (ranges[i][1]-ranges[i][0]) + \
            ranges[i][0] for i in range(len(rows[0]))]


# Kcluster with restart polices
def kcluster(rows, distance=euclidean, k=4, trials=100, groups=5, beta=100):
    # Lo important es tenir un centroide amb molts items i a poca distancia d'ells
    # Beta -> relacio optima distancia/num_items
    iters_per_alg = trials / groups
    min_td = sys.maxint
    best_clusters_assign = None

    last_clusters = []

    # Generate k centroides
    for _ in range(k):
        last_clusters.append(generate_random_clusters(rows))

    for _ in xrange(groups):
        td, clusters_items, clusters_dist = _kcluster(rows, distance, k, iters_per_alg, clusters=copy.deepcopy(last_clusters))

        if td < min_td:
            min_td = td
            best_clusters_assign = clusters_items

        # If a centroid is considered as good -> no reason for randomize it
        min_dist_clusters = map(lambda x: x[0],
                                filter(lambda x : len(clusters_items[x[0]]) != 0 and x[1] / len(clusters_items[x[0]]) <= beta,
                                enumerate(clusters_dist)))
        for i in range(k):
            if i not in min_dist_clusters:
                last_clusters[i] = generate_random_clusters(rows)

    return min_td, best_clusters_assign


# Generate graphic total distance as function of k
def total_distance_k(rows, init_k=2, end_k=20, restart_polices=True, path='distance_as_function_k.txt'):
    with open(path, 'w') as f:
        f.write("#Number of clusters\tTotal distance")
        for i in range(init_k, end_k + 1):
            if restart_polices:
                td, _ = kcluster(rows, k=i)
            else:
                td, _, _ = _kcluster(rows, k=i)
            f.write(str(i) + "\t" + "{0:.2f}".format(td) + "\n")

if __name__ == "__main__":
    bn, w, d = readfile("blogdata.txt")
    total_distance_k(d)
    total_distance_k(d, restart_polices=False, path="distance_as_function_k_nrp.txt")
