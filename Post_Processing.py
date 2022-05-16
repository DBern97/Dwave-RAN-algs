"""
Contains methods for handling the anneal data.
"""

class Try:
    def __init__(self):

    def nothing(x):
        return  0

    def Make_Graph_From_Qubo(qubo):
        qubo = np.array(qubo, dtype=float)

        G = nx.from_numpy_matrix(qubo)
        nx.draw(G, with_labels=True)
        return G

    def Plot_Energies(res):
        return []

    def Sub_Back(config):
        return []

    def Test(x):
        return x
