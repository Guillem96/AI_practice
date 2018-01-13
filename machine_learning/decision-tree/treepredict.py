#!/usr/bin/env python

import argparse
import collections
import itertools
import math
import sys
import time
import random
from decisionnode import decisionnode

# ---- t3 ----

def read_file(file_path, data_sep=' ', ignore_first_line=False):
    with open(file_path, 'r') as f:
        return read_stream(f, data_sep, ignore_first_line)


def read_stream(stream, data_sep=' ', ignore_first_line=False):
    strip_reader = (l.strip()  for l in stream)
    filtered_reader = (l for l in strip_reader if l)
    start_at = 1 if ignore_first_line else 0

    prototypes = []
    for line in itertools.islice(filtered_reader, start_at, None):
        tokens = itertools.imap(str.strip, line.split(data_sep))
        prototypes.append(map(filter_token, tokens))

    return prototypes


def filter_token(token):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


# ---- t4 ----

def unique_counts(part):
    # return collections.Counter(row[-1] for row in part)
    counts = {}
    for row in part:
        counts[row[-1]] = counts.get(row[-1], 0) + 1
    return counts


# ---- t5 ----
def gini_impurity(part):
    total = float(len(part))
    counts = unique_counts(part)

    probs = (v / total for v in counts.itervalues())
    return 1 - sum((p * p for p in probs))

# ---- t6 ----
def entropy(rows):
    from math import log
    # Define log2
    log2 = lambda x: log(x)/log(2)

    results = unique_counts(rows)
    total = float(len(rows))

    probs = (v / total for v in results.itervalues())

    return -sum((p * log2(p) for p in probs))

# ---- t7 ----
def divideset(part, column, value):
    def split_num(prot): return prot[column] >= value
    def split_str(prot): return prot[column] == value

    split_fn = split_num if isinstance(value, (int, float)) else split_str

    set1, set2 = [], []
    for prot in part:
        s = set1 if split_fn(prot) else set2
        s.append(prot)

    return set1, set2


def imp_increment(set1, set2, imp_function=gini_impurity):
    p1 = float(len(set1)) / float(len(set1) + len(set2))
    p2 = float(len(set2)) / float(len(set1) + len(set2))

    return imp_function(set1 + set2) - \
            p1 * imp_function(set1) - \
            p2 * imp_function(set2)

# Donat un set d items retorna la seva millor particio
def get_best_partition(part, imp_function=gini_impurity):
    # Set up some variables to track the best criteria
    best_gain = 0.0     # Best imp increment
    criteria = None     # Best (value, col)
    best_sets = None    # Best results

    for index in range(len(part[0]) - 1):
        values = set(map(lambda v: v[index], part))
        for value in values:
            true_set, false_set = divideset(part, index, value)
            gain = imp_increment(true_set, false_set, imp_function)

            if best_gain < gain and len(true_set) > 0 and len(false_set) > 0:
                criteria = (value, index)
                best_sets = true_set, false_set
                best_gain = gain

    return criteria, best_sets

# ---- t9 ----
def buildtree(part, scoref=gini_impurity, beta=0):
    if len(part) == 0: return decisionnode()

    current_score = scoref(part)

    if(current_score > beta):
        # get best criteria
        criteria, best_sets = get_best_partition(part, scoref)

        if criteria:
            # Repeat the action for left and right set
            true_tree = buildtree(best_sets[0], scoref, beta)
            false_tree = buildtree(best_sets[1], scoref, beta)
            return decisionnode(col=criteria[1], value=criteria[0], tb=true_tree, fb=false_tree)
        else:
            return decisionnode(results=unique_counts(part))
    else:
        return decisionnode(results=unique_counts(part))

# ---- t10 ----
def buildtree_iterative(part, scoref=gini_impurity, beta=0):

    # Stack that will contains the (set, tree position)
    fringe = []

    # Initial partition
    criteria, best_sets = get_best_partition(part, scoref)

    # Set the root node
    fb = decisionnode(answer=False)
    tb = decisionnode(answer=True)
    root = decisionnode(col=criteria[1], value=criteria[0], tb=tb, fb=fb)

    fb.parent = root
    tb.parent = root

    fringe.append(fb) # False
    fringe.append(tb) # True

    while len(fringe) != 0:
        current_node = fringe.pop()
        current_partition = get_node_partition(current_node, part)
        current_score = scoref(current_partition)

        if current_score > beta:
            criteria, best_sets = get_best_partition(current_partition, scoref)
            if criteria:
                current_node.col=criteria[1]
                current_node.value=criteria[0]

                # If false
                fb = decisionnode(parent=current_node, answer=False)
                current_node.fb = fb
                fringe.append(fb) # False

                # if true
                tb = decisionnode(parent=current_node, answer=True)
                current_node.tb = tb
                fringe.append(tb) # True
            else:
                current_node.results=unique_counts(current_partition)

        else:
            current_node.results=unique_counts(current_partition)


    return root

def get_node_partition(node, source_partition):
    answer_list = [node.answer] # Will work as stack
    parent = node.parent

    # get parent and answers to get to him
    while parent.parent != None:
        answer_list.append(parent.answer)
        parent = parent.parent

    partition = source_partition

    # generate partition
    while len(answer_list) != 0:
        true_set, false_set = divideset(partition, parent.col, parent.value)
        if answer_list.pop():
            partition = true_set
            parent = parent.tb
        else:
            partition = false_set
            parent = parent.fb

    return partition

# ---- t11 ----
def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print str(tree.col) + ':' + str(tree.value) + '?'
        # Print the branches
        print indent + 'True->',
        printtree(tree.tb, indent + '  ')
        print indent + 'False->',
        printtree(tree.fb, indent + '  ')

# ---- t12 ----
def classify(obj, tree):
    current_node = tree
    def split_num(prot): return prot[current_node.col] >= current_node.value
    def split_str(prot): return prot[current_node.col] == current_node.value

    while not current_node.is_leaf():
        split_fn = split_num if isinstance(current_node.value, (int, float)) else split_str
        if split_fn(obj):
            current_node = current_node.tb
        else:
            current_node = current_node.fb

    return current_node.results

def roulette(results, seed=time.time()):
    total = float(sum(results.itervalues()))
    probs = [results[results.keys()[0]] / total]
    
    for k in results.keys()[1:-1]:
        probs.append(probs[-1] + (results[k] / total))

    probs.append(1.0)

    random.seed(seed)
    rand = random.random()

    for i, p in enumerate(probs):
        if p >= rand:
            return results.keys()[i]
def test_preformance(testset,trainingset,beta=0,trials=50,testprob=0.2):
    error=0.0
    tree = cross_validate(trainingset,beta,trials,testprob)
    for prot in testset:
        results = classify(prot[:-1],tree)
        tree_result = roulette(results)
        real_result = prot[-1]
        if tree_result != real_result:
            error+=1  
    return error/len(testset)

def cross_validate(trainingset, beta, trials, testprob):
    best_tree=None
    min_error=1.0
    for i in xrange(trials):
        trainset, testset = divide_data(trainingset,testprob)
        error,tree= test_set(testset,trainingset,beta)
        if error == 0.0: return tree #Estalviar temps   
        if error < min_error:
            min_error=error
            best_tree=tree
    return best_tree 

def test_set(testset, trainingset, beta):
    if len(testset)==0 : return 1, None
    tree = buildtree(trainingset,beta=beta)
    error=0.0
    for prot in testset:
        results = classify(prot[:-1],tree)
        tree_result = roulette(results)
        real_result = prot[-1]
        if tree_result != real_result:
            error+=1
    
    return error/len(testset), tree

def divide_data(data, testprob):
    random.seed(time.time())
    training = []
    test = []
    for i in data:
        if random.random()<testprob:
            test.append(i)
        else:
            training.append(i)
    return training, test
# ---- t15 ----
def fill_missingdata(data,missingvalue=None):
    col_info=[] # [(type, median/prob_list),...]
    for i in xrange(len(data[0])-1): #-1 to avoid solution
        col_list = map(lambda r: r[i], data)
        col_type=find_type(col_list,missingvalue)
        if col_type == 'str':
            info=md_uniquecounts(col_list,missingvalue)
        else:
            info=median(col_list,missingvalue)
        col_info.append((col_type,info))

    for row, prot in enumerate(data):
        for col, value in enumerate(prot):
            if value == missingvalue:
                if col_info[col][0] == 'str':
                    data[row][col]=roulette(col_info[col][1])
                else:
                    data[row][col]=col_info[col][1]
#Busca el tipo de la lista
def find_type(data_list,missingvalue):
    for value in data_list:
         if not isinstance(value, (int, float)) and value != missingvalue:
                return 'str'
    return 'num' 
#Cuenta el numero de cada valor en la lista
def md_uniquecounts(data_list,missingvalue):
    counts={}
    for value in data_list:
        if value != missingvalue:
            counts[value] = counts.get(value, 0) + 1
    return counts
#La mediana de la lista (contando que es una lista de numeros)
def median(data_list,missingvalue):
    filtered_data=list(filter(lambda x: x != missingvalue, data_list))
    return filtered_data[len(filtered_data)/2]


# ---- t16 ----
def prune(tree, beta):
    if not tree.tb.is_leaf():
        prune(tree.tb, beta)

    if not tree.fb.is_leaf():
        prune(tree.fb, beta)

    if tree.tb.is_leaf() and tree.fb.is_leaf():
        # Build a combined dataset
        tb, fb = [], []
        for v, c in tree.tb.results.items( ):
            tb += [[v]] * c
        for v, c in tree.fb.results.items( ):
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = imp_increment(tb, fb, entropy)
        if delta < beta:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)
# ------------------------ #
#        Entry point       #
# ------------------------ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="---- DESCRIPTION HERE ----",
        epilog="---- EPILOG HERE ----")

    parser.add_argument('prototypes_file', type=argparse.FileType('r'),
                        help="File filled with prototypes (one per line)")

    parser.add_argument('-ifl', '--ignore_first_line', action='store_true',
                        help="Ignores the first line of the prototypes file")

    parser.add_argument('-ds', '--data_sep', required=False, default=',',
                        help="Prototypes data fields separation mark")

    parser.add_argument('-s', '--seed', default=int(time.time()), type=int,
                        help="Random number generator seed.")

    options = parser.parse_args()

    # Example code
    protos = read_stream(options.prototypes_file, options.data_sep, True)

    # **** Your code here ***
    training, test = divide_data(protos,0.2)
    error_prob=test_preformance(training,test,0.3,100,0.2)
    print 'Test preformance: ' + str(float('{0:.2f'.format.(1-error_prob)*100)) + '%'
    fill_missingdata(protos,'5more')
