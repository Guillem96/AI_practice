#!/usr/bin/env python

import argparse
import collections
import itertools
import math
import sys
import time
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

# ---- t9 ----
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

"""
    beta: Impuresa amb la qual estem satisfets en una fulla
"""
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


def buildtree_iterative_w_nodes(part, scoref=gini_impurity, beta=0):

    # Stack that will contains the (set, tree position)
    fringe = []

    # Initial partition
    criteria, best_sets = get_best_partition(part, scoref)

    # Set the root node
    fb = decisionnode(part=best_sets[1])
    tb = decisionnode(part=best_sets[0])
    root = decisionnode(col=criteria[1], value=criteria[0], tb=tb, fb=fb)

    fringe.append(fb) # False
    fringe.append(tb) # True

    while len(fringe) != 0:
        current_node = fringe.pop()

        current_score = scoref(current_node.part)

        if current_score > beta:
            criteria, best_sets = get_best_partition(current_node.part, scoref)

            current_node.col=criteria[1]
            current_node.value=criteria[0]

            # If false
            fb = decisionnode(part=best_sets[1])
            current_node.fb = fb
            fringe.append(fb) # False

            # if true
            tb = decisionnode(part=best_sets[0])
            current_node.tb = tb
            fringe.append(tb) # True
        else:
            current_node.results=unique_counts(current_node.part)


    return root


def buildtree_iterative_l_nodes(part, scoref=gini_impurity, beta=0):

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


    return root

def get_node_partition(node, source_partition):
    answer_list = [node.answer] # Will work as stack
    parent = node.parent

    # get parent and answers to get to him
    while True:
        if parent.parent == None: break
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

def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print str(tree.col) + ':' + str(tree.value) + ' ? '
        # Print the branches
        print indent + 'True-> ',
        printtree(tree.tb, indent + '    ')
        print indent + 'False-> ',
        printtree(tree.fb, indent + '    ')

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
    for p in protos:
        print p

    print unique_counts(protos)

    # **** Your code here ***
    printtree(buildtree_iterative_l_nodes(protos))
