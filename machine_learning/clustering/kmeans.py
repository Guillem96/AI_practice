import random
import copy

def kcluster(rows,distance=pearson,k=4):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),
            max([row[i] for row in rows])) for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    clusters=[[random.random() * (ranges[i][1]-ranges[i][0]) + \
                ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]

    lastmatches = None
    for t in range(100):
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
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    return bestmatches
