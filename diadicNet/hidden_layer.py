from numpy         import ndarray

from hidden_neuron import HiddenNeuron
from input_layer   import InputLayer

class HiddenLayer:
   
   def __init__(
                self, 
                input_to_hidden_map: dict[tuple[int], list[tuple[int]]],
                activation_function: str,
                input_layer:         InputLayer       
               
                ) -> None:
    
    ''' Initiate neurons in hidden layer '''

    self.neuron_list         = {}
    self.source_layer        = input_layer
    self.activation_funtion  = activation_function
    
    for neuron_id, input_ids in input_to_hidden_map.items():

      self.input_neurons          = [self.source_layer[id] for id in input_ids]                # get corresponding input neurons
      self.neuron_list[neuron_id] =  HiddenNeuron(self.input_neurons, self.activation_funtion)


   
   def __call__(self, weigts: dict[tuple[int], ndarray]) -> None:
      
      for neuron_id in self.neuron_list.keys():
        
        neuron         = self.neuron_list[neuron_id]
        neuron_weights = weigts[neuron_id]

        neuron(neuron_weights)

  
   def __getitem__(self, neuron_id: tuple[int] ) -> HiddenNeuron:
       '''Get neuron by its id.'''
       return self.neuron_list[neuron_id]
   

