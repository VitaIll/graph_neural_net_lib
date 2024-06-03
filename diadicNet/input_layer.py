from numpy             import ndarray, array

from feature_extractor import FeatureExtractor
from input_neuron      import InputNeuron


class InputLayer:
    ''' 
        Input layer of the neural network.
          - for now assumes output for a single macro unit, e.g. one firm/ state etc..
          - network partition is assumed to be repressented by list of sub_matricies,
            we can also consider alternative containers.
          - neuron list updated to dictionary
    '''
    
    def __init__(self, neuron_ids: list[tuple[int]],  feature_list: list[str]) -> None:
          ''' 
              Initiate input layer by a passing neuron ids list,
              with elements stored as [(m,n)].
          '''
          self.feature_list      = feature_list
          self.feature_extractor = FeatureExtractor(feature_list)                # create reusable feature extractor instance
          self.neuron_ids        = neuron_ids                                    # store network partition
          
          self.neuron_list = {}

          for id in neuron_ids:
               self.neuron_list[id] = InputNeuron(id, self.feature_extractor)   # initiate input neurons

    
    def __call__(self, network_partition: dict[tuple[int], ndarray[int]]) -> None:
         ''' Pass data to the input layer. '''

         for id, neuron in self.neuron_list.items():
              adj_matrix = network_partition[id]
              neuron(adj_matrix)
    
    
    def __getitem__(self, id: tuple[int]) -> InputNeuron:
         ''' Get fetures from neuron identified by the (m,n) tuple id.'''
         return self.neuron_list[id]
        