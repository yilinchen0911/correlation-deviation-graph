#!/usr/bin/env python3
import operator
'''
    Implements the signed Louvain method.
    Input: a signed weighted undirected graph
    Ouput: a (partition, modularity) pair where modularity is maximum
'''


class SignedLouvain:

    '''
        Builds a graph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''
    @classmethod
    def from_file(cls, path, sign):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes[int(n[0])] = 1
            nodes[int(n[1])] = 1
            if len(n) == 3:
                if sign == "positive":
                    w = float(n[2])
                elif sign == "negative":
                    w = -float(n[2])
                else:
                    raise Exception(
                        "argument sign should be either positive or negative")
                wP = max(0, w)
                wN = max(0, -w)
            edges.append(((int(n[0]), int(n[1])), (wP, wN)))
        # rebuild graph with successive identifiers
        nodes_, edges_ = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
        return cls(nodes_, edges_)

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        # precompute m (sum of the weights of all links in network)
        #            k_i (sum of the weights of the links incident to node i)
        self.mP = 0
        self.mN = 0
        self.k_iP = [0 for n in nodes]
        self.k_iN = [0 for n in nodes]
        self.edges_of_node = {}
        self.wP = [0 for n in nodes]
        self.wN = [0 for n in nodes]
        for e in edges:
            self.mP += e[1][0]
            self.mN += e[1][1]
            self.k_iP[e[0][0]] += e[1][0]
            self.k_iP[e[0][1]] += e[1][0]
            self.k_iN[e[0][0]] += e[1][1]
            self.k_iN[e[0][1]] += e[1][1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # access community of a node in O(1) time
        self.communities = [n for n in nodes]
        self.actual_partition = []

    '''
        Applies the Louvain method.
    '''

    def apply_method(self):
        network = (self.nodes, self.edges)
        best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        while 1:
            # print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network)
            q = self.compute_modularity(partition)
            partition = [c for c in partition if c]
            print((partition, q))
            #print("%s (%.8f)" % (partition, q))
            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            if abs(q - best_q) < 1E-9:
                break
            network = self.second_phase(network, partition)
            best_partition = partition
            best_q = q
        return (self.actual_partition, best_q)

    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''

    def compute_modularity(self, partition):
        qP = 0
        qN = 0
        m2P = self.mP * 2
        m2N = self.mN * 2
        for i in range(len(partition)):
            if m2P > 0:
                qP += self.s_inP[i] / m2P - (self.s_totP[i] / m2P) ** 2
            # print self.s_inP[i], m2P, self.s_totP[i]
            if m2N > 0:
                qN += self.s_inN[i] / m2N - (self.s_totN[i] / m2N) ** 2
        return (m2P * qP - m2N * qN) / (m2P + m2N)

    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''

    def compute_modularity_gain(self, node, c, k_i_inP, k_i_inN):
        # print self.mP, k_i_inP, self.s_totP, self.k_iP
        if self.mP == 0:
            deltaQP = 0
        else:
            deltaQP = 1 / (2 * self.mP) * (k_i_inP -
                                           self.s_totP[c] * self.k_iP[node] / self.mP)
        if self.mN == 0:
            deltaQN = 0
        else:
            deltaQN = 1 / (2 * self.mN) * (k_i_inN -
                                           self.s_totN[c] * self.k_iN[node] / self.mN)
        return (2 * self.mP * deltaQP - 2 * self.mN * deltaQN) / (2 * self.mP + 2 * self.mN)

    '''
        Performs the first phase of the method.
        _network: a (nodes, edges) pair
    '''

    def first_phase(self, network):
        # make initial partition
        best_partition = self.make_initial_partition(network)
        while 1:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                # default best community is its own
                best_community = node_community
                best_gain = 0
                # remove _node from its community
                best_partition[node_community].remove(node)
                best_shared_linksP = 0
                best_shared_linksN = 0
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_linksP += e[1][0]
                        best_shared_linksN += e[1][1]
                self.s_inP[node_community] -= 2 * \
                    (best_shared_linksP + self.wP[node])
                self.s_inN[node_community] -= 2 * \
                    (best_shared_linksN + self.wN[node])
                self.s_totP[node_community] -= self.k_iP[node]
                self.s_totN[node_community] -= self.k_iN[node]
                self.communities[node] = -1
                communities = {}  # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    if community in communities:
                        continue
                    communities[community] = 1
                    shared_linksP = 0
                    shared_linksN = 0
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if e[0][0] == node and self.communities[e[0][1]] == community or e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_linksP += e[1][0]
                            shared_linksN += e[1][1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(
                        node, community, shared_linksP, shared_linksN)
                    if gain > best_gain:
                        best_community = community
                        best_gain = gain
                        best_shared_linksP = shared_linksP
                        best_shared_linksN = shared_linksN
                # insert _node into the community maximizing the modularity gain
                # print self.s_inP
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                self.s_inP[best_community] += 2 * \
                    (best_shared_linksP + self.wP[node])
                self.s_inN[best_community] += 2 * \
                    (best_shared_linksN + self.wN[node])
                self.s_totP[best_community] += self.k_iP[node]
                self.s_totN[best_community] += self.k_iN[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    '''
        Yields the nodes adjacent to _node.
        _node: an int
    '''

    def get_neighbors(self, node):
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]:  # a node is not neighbor with itself
                continue
            if e[0][0] == node:
                yield e[0][1]
            if e[0][1] == node:
                yield e[0][0]

    '''
        Builds the initial partition from _network.
        _network: a (nodes, edges) pair
    '''

    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_inP = [0 for node in network[0]]
        self.s_inN = [0 for node in network[0]]
        self.s_totP = [self.k_iP[node] for node in network[0]]
        self.s_totN = [self.k_iN[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]:  # only self-loops
                self.s_inP[e[0][0]] += e[1][0]
                self.s_inN[e[0][0]] += e[1][1]
                self.s_inP[e[0][1]] += e[1][0]
                self.s_inN[e[0][1]] += e[1][1]
        return partition

    '''
        Performs the second phase of the method.
        _network: a (nodes, edges) pair
        _partition: a list of lists of nodes
    '''

    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]
        # relabelling communities
        communities_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        self.communities = communities_
        # building relabelled edges
        edges_ = {}
        for e in network[1]:
            ci = self.communities[e[0][0]]
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] = tuple(
                    map(operator.add, edges_[(ci, cj)], e[1]))
            except KeyError:
                edges_[(ci, cj)] = e[1]
        edges_ = [(k, v) for k, v in edges_.items()]
        # recomputing k_i vector and storing edges by node
        self.k_iP = [0 for n in nodes_]
        self.k_iN = [0 for n in nodes_]
        self.edges_of_node = {}
        self.wP = [0 for n in nodes_]
        self.wN = [0 for n in nodes_]
        for e in edges_:
            self.k_iP[e[0][0]] += e[1][0]
            self.k_iN[e[0][0]] += e[1][1]
            self.k_iP[e[0][1]] += e[1][0]
            self.k_iN[e[0][1]] += e[1][1]
            if e[0][0] == e[0][1]:
                self.wP[e[0][0]] += e[1][0]
                self.wN[e[0][0]] += e[1][1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)


'''
    Rebuilds a graph with successive nodes' ids.
    _nodes: a dict of int
    _edges: a list of ((int, int), weight) pairs
'''


def in_order(nodes, edges):
    # rebuild graph with successive identifiers
    nodes = list(nodes.keys())
    nodes.sort()
    i = 0
    nodes_ = []
    d = {}
    for n in nodes:
        nodes_.append(i)
        d[n] = i
        i += 1
    edges_ = []
    for e in edges:
        edges_.append(((d[e[0][0]], d[e[0][1]]), (e[1][0], e[1][1])))
    return (nodes_, edges_)
