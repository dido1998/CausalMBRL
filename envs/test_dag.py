import networkx as nx
import numpy as np
import torch



"""
def random_dag(nodes, edges):
    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = np.random.randint(0,nodes)
        b=a
        while b==a:
            b = np.random.randint(0,nodes)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G



if __name__ == '__main__':
    edges = (5 * (5 - 1))//2
    g = random_dag(5, np.random.randint(1, edges + 1))
    adj_matrix = nx.to_numpy_matrix(g)
    print(adj_matrix)"""


M = 5
N = 5

expParents = 5
idx        = np.arange(M).astype(np.float32)[:,np.newaxis]
print(idx)
idx_maxed  = np.minimum(idx, expParents)
print(idx_maxed)
p          = np.broadcast_to(idx_maxed/(idx+1), (M, M))
print(p)
p          = np.tril(p, -1)
print(p)
B          = np.random.binomial(1, p)
print(B)






"""
gammagt = np.zeros((M, M))            
g = '0->1, 0->2, 1->2'
for e in g.split(","):
    if e == "": continue
    nodes = e.split("->")
    if len(nodes) <= 1: continue
    nodes = [int(n) for n in nodes]
    for src, dst in zip(nodes[:-1], nodes[1:]):
        if dst > src:
            gammagt[dst,src] = 1
        elif dst == src:
            raise ValueError("Edges are not allowed from " +
                             str(src) + " to oneself!")
        else:
            raise ValueError("Edges are not allowed from " +
                             str(src) + " to ancestor " +
                             str(dst) + " !")

print(gammagt)"""