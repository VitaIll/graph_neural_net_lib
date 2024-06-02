import networkx 

from   numpy import ndarray
from   utils import Utils, Reference


class FeatureExtractor:
    ''' 
        Feature extractor computes desired features from the adjacency matrix passed as an argument.
        - from v 0.0.2 the feature extractor is initiated as empty class and features are computed
           using the call operator and list of network statistics as an argument 
        - it is crucial to comply with networks library API terminology !!!
    '''
    
    def __init__(self, network_stats: list[str]) -> None:
        
        for stat in network_stats:
            
            self.valid_stats = []

            if Utils.function_exists(networkx, stat):       # check that networkx indeed contains desired function
                
                self.valid_stats.append(stat)               # store name of the function in the list

            else:
                raise ValueError(f"{stat} is not defined in networkx library, check documentation at {Reference.networkx_link}.")
     

    
    def __call__(self, adjacency_matrix: ndarray[int]) -> dict[tuple[int],float]:
       
        graph = networkx.from_numpy_array(adjacency_matrix)  # convert to networkx native type
        res   = {}
        
        for stat in self.valid_stats:                        # eval network statistics for all valid names in the list
            
            method    = getattr(networkx, stat)
            res[stat] = method(graph)
                                                  

