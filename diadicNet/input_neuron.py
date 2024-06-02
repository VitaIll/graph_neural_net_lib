
from   typing              import Union
from   numpy               import ndarray

from   feature_extractor   import FeatureExtractor

class InputNeuron:
    ''' 
        Neuron of the input layer.
          - is identified by a duplet (i,j) and relevant part 
            of the adjacency matrix
          - contain features extracted from the sub-matrix defining 
            conections between worker of firm i and j.
    '''
    
    def __init__(self, neuron_id: tuple[int], feature_extractor: FeatureExtractor) -> None:
        ''' Initiate neuron with id (m,n) tuple, extractor instance and list of required features.'''
        self.id            = neuron_id
        self.extractor     = feature_extractor
    
   
    def __call__(self, adjacency_matrix: ndarray[int]) -> None:
        ''' Compute fetures from partial adjacency matrix.'''
        self.features = self.extractor(adjacency_matrix)
  
    
    def __getitem__(self, feature_name: str) -> Union[float, ndarray]:
        ''' Retrieve feature by its name.'''
        return self.features[feature_name]
