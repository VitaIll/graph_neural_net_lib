
import numpy as np 
from hidden_neuron import HiddenNeuron

class HiddenLayer:
   
   def __init__(
                self, 
                unit_idx:            int,
                unit_count:          int,
                activation_function: str,
                ) -> None:
    
    self.neuron_list = []

    for idx in range(unit_count):
      
      neuron_idx = tuple(unit_idx, idx)
      neuron     = HiddenNeuron(neuron_idx, activation_function)
      
      self.neuron_list.append(neuron)
   
