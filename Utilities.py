"""
Contains helper functions to read, plot and analyse anneal data.
"""


import networkx as nx

def Make_Graph_From_Qubo(qubo):
    qubo = np.array(qubo, dtype=float)

    G = nx.from_numpy_matrix(qubo)
    nx.draw(G, with_labels=True)
    return G

def Plot_Energies(res):
    return []

def Sub_Back(config):
    return []