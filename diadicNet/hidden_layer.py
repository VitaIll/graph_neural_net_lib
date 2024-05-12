
import numpy as np 

from numpy import ndarray

from hidden_neuron import HiddenNeuron
from input_layer   import InputLayer

class HiddenLayer:
   
   def __init__(
                self, 
                unit_idx:            int,
                unit_count:          int,
                activation_function: str,
                input_layer: InputLayer       
               
                ) -> None:
    
    self.neuron_list  = {}
    
    for idx in range(1,unit_count):
      
      neuron_idx   = (unit_idx, idx)
      feature_list = input_layer[neuron_idx]

      if neuron_idx == (unit_idx, unit_idx):
        pass
      else:
        feature_list += input_layer[(unit_idx, unit_idx)]

      neuron            = HiddenNeuron(neuron_idx, feature_list, activation_function)
      neuron_index_pair = {neuron_idx: neuron}

      self.neuron_list.update(neuron_index_pair)

   
   def __call__(self, weigts: dict) -> None:
      
      for neuron_id in self.neuron_list.keys():
        
        neuron         = self.neuron_list[neuron_id]
        neuron_weights = weigts(neuron_id)

        neuron(neuron_weights)

  
   def __getitem__(self, neuron_id: tuple ) -> tuple:
     neuron = self.neuron_list[neuron_id]

     return (neuron.weights, neuron.features, neuron.output_value)
  
   

