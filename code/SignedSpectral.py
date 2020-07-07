import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def in_order(nodes, edges):
    # rebuild graph with successive identifiers
    nodes = list(nodes)
    nodes.sort()
    i = 0
    nodes_ = set()
    d = {}  # key original id value new id from 0
    for n in nodes:
        nodes_.add(i)
        d[n] = i
        i += 1
    edges_ = {}
    for e in edges.items():
        edges_[(d[e[0][0]], d[e[0][1]])] = e[1]
    return (nodes_, edges_, d)


class SignedSpectral:
    def __init__(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = set()
        edges = {}
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes.add(int(n[0]))
            nodes.add(int(n[1]))
            if len(n) == 3:
                w = float(n[2])
            edges[(int(n[0]), int(n[1]))] = w
            edges[(int(n[1]), int(n[0]))] = w
        self.nodes, self.edges, self.dict = in_order(nodes, edges)
        # build adjacency matrix
        self.A = np.zeros((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            self.A[i, i] = 0
            for j in range(len(self.nodes)):
                if (i, j) in self.edges:
                    self.A[i, j] = self.edges[(i, j)]

    def cluster(self, L_bar, num_eig, num_clust):
        w, v = eigh(L_bar, eigvals=(0, num_eig - 1))
        kmeans = KMeans(n_clusters=num_clust, random_state=0).fit(v)
        partition = [[] for _ in range(num_clust)]
        for i, label in enumerate(kmeans.labels_):
            partition[label].append(i)
        return partition

    def apply_method(self, num_clust=20, num_eig=10, method="normalizingVertices"):
        D_bar = np.diag(np.sum(np.absolute(self.A), axis=1).reshape(-1))
        if method == "normalizingVertices":
            L_bar = D_bar - self.A
        if method == "normalizingVolume":
            L_bar = np.matmul(np.linalg.inv(D_bar), self.A)
        partition = self.cluster(L_bar, num_eig, num_clust)
        return partition

    def cluster_by_first_eigenvector(self, cutQuantile=0.5):
        return self.apply_method(num_clust=2, num_eig=1)

    def cluster_by_second_eigenvector(self, cutQuantile=0.5):
        return self.apply_method(num_clust=2, num_eig=2)
