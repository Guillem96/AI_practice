#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools  # May be useful ;)
import sys

import wcnf


class Graph(object):

    def __init__(self):
        self.edges = []
        self.num_nodes = 0

    def read_file(self, file_path):
        with open(file_path, 'r') as stream:
            self.read_stream(stream)

    def read_stream(self, stream):
        num_edges = -1

        reader = (l for l in (ll.strip() for ll in stream) if l)
        for line in reader:
            l = line.split()
            if l[0] == 'p':
                self.num_nodes = int(l[2])
                num_edges = int(l[3])
            elif l[0] == 'c':
                pass  # Ignore comments
            else:
                self.edges.append([int(l[1]), int(l[2])])

        if num_edges != len(self.edges):
            print "Warning incorrect number of edges"

    def min_vertex_cover_to_maxsat(self):
        msat = wcnf.WCNFFormula()

        # Soft: Including a vertex in the cover has a cost of 1
        for _ in xrange(self.num_nodes):
            msat.add_clause([-msat.new_var()], 1)  # clause weight = 1

        # Hard: All edges must be covered
        for n1, n2 in self.edges:
            msat.add_clause([n1, n2], 0)  # 0 weight means top
        return msat

    def max_clique_to_maxsat(self):
        msat = wcnf.WCNFFormula()

        # Soft: Including a vertex in the cover has a cost of 1
        for _ in xrange(self.num_nodes):
            msat.add_clause([msat.new_var()], 1)  # clause weight = 1

        for x in range(1, self.num_nodes):
            for y in range(x + 1, self.num_nodes + 1):
                if [x, y] not in self.edges and [y, x] not in self.edges:
                    msat.add_clause([-x, -y], 0)  # 0 weight means top
        return msat

    def max_cut_to_maxsat(self):
        msat = wcnf.WCNFFormula()
        # **** Your code here ****

        return msat


if __name__ == '__main__':
    if len(sys.argv) > 1:
        g = Graph()
        g.read_file(sys.argv[1])

        mvcovermsat = g.min_vertex_cover_to_maxsat()
        mvcovermsat.write_dimacs_file('mvcover.wcnf')
        # mvcovermsat.to_1_3().write_dimacs_file('mvcover_1_3.wcnf')

        mcliquemsat = g.max_clique_to_maxsat()
        mcliquemsat.write_dimacs_file('mclique.wcnf')
        # mcliquemsat.to_1_3().write_dimacs_file('mclique_1_3.wcnf')

        mcutmsat = g.max_cut_to_maxsat()
        mcutmsat.write_dimacs_file('mcut.wcnf')
        # mcutmsat.to_1_3().write_dimacs_file('mcut_1_3.wcnf')
