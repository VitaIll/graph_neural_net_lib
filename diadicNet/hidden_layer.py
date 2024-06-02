
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
                input_layer:         InputLayer       
               
                ) -> None:
    
    self.neuron_list                = {}
    self.source_layer               = input_layer
    self.unit_count                 = unit_count
    self.unit_id                    = unit_idx
    self.activation_funtion         = activation_function
    
    for idx in range(1,unit_count):
      
      neuron_idx   = (unit_idx, idx)
      feature_list = (input_layer[neuron_idx])['features']

      if neuron_idx == (unit_idx, unit_idx):
        pass
      else:
        feature_list += (input_layer[(unit_idx, unit_idx)])['features']
        
      neuron            = HiddenNeuron(neuron_idx, feature_list, activation_function)
      neuron_index_pair = {neuron_idx: neuron}

      self.neuron_list.update(neuron_index_pair)

   
   def __call__(self, weigts: dict) -> None:
      
      for neuron_id in self.neuron_list.keys():
        
        neuron         = self.neuron_list[neuron_id]
        neuron_weights = weigts(neuron_id)

        neuron(neuron_weights)

  
   def __getitem__(self, neuron_id: tuple|None ) -> tuple:
       neuron = self.neuron_list[neuron_id]

       return neuron.properties
   
   
   def __update(self):
      
      for idx in range(1,self.unit_count):
      
        neuron_idx    = (self.unit_id, idx)
        feature_list  = (self.source_layer[neuron_idx])['features']

        if neuron_idx == (self.unit_id, self.unit_id):
          pass
        else:
          feature_list += (self.source_layer[(self.unit_id, self.unit_id)])['features']
        
        neuron            = HiddenNeuron(neuron_idx, feature_list, activation_function)
        neuron_index_pair = {neuron_idx: neuron}

        self.neuron_list.update(neuron_index_pair)
   
   
   def get_values(self) -> ndarray:
     
     vals  = []

     for neuron in self.neuron_list.values():
       vals.append(neuron.output_value)

     return np.array(vals)
       
  
   

