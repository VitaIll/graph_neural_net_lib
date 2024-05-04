
import numpy               as np
from   feature_extractor   import FeatureExtractor

class InputNeuron:
    ''' 
        Neuron of the input layer.
          - is identified by a duplet (i,j) and relevant part 
            of the adjacency matrix
          - contain features extracted from the sub-matrix defining 
            conections between worker of firm i and j.
    '''
    def __init__(self, idx_tuple: tuple ,sub_matrix: np.ndarray) -> None:
          self.idx_tuple           = idx_tuple
          self.m_feature_extractor = FeatureExtractor(sub_matrix)
 