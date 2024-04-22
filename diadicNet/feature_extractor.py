import networkx as nx
import numpy    as np

class FeatureExtractor:
    ''' 
        Feature extractor computes desired features from the adjacency matrix passed as an argument.
    '''
    def __init__(self, adj_matrix: np.ndarray) -> None:
        
        self.m_feature_list   = []

        self.m_centrality     = FeatureExtractor.centrality      (adj_matrix)
        self.m_clustering     = FeatureExtractor.clustering_coef (adj_matrix)

        self.m_feature_list.append (self.m_centrality)
        self.m_feature_list.append (self.m_clustering)

    
    ''' Method computing degree centrality.''' 

    def centrality (adj_matrix: np.ndarray) -> float:
        
        G = nx.from_numpy_array(adj_matrix) 
        
        node_dictionary = nx.degree_centrality (G)
        centrality_vals = list(node_dictionary.values())
        centrality_vals = np.array(centrality_vals)

        return centrality_vals.mean()
    
    
    ''' Method computing average clustering coefficient.''' 

    def clustering_coef (adj_matrix: np.ndarray) -> float:
        
        G = nx.from_numpy_array(adj_matrix) 

        return nx.average_clustering (G)
        