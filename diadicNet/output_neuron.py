import numpy as np

from numpy  import ndarray
from typing import Optional

from hidden_layer import HiddenLayer

class OutputNeuron:

   def __init__(self, input_vals: ndarray, hidden_layer: HiddenLayer) -> None:
      
      self.__hidden_layer = hidden_layer

      self.__dim     = len(input_vals)
      self.__vals    = input_vals
      self.__weights = np.random.random(self.__dim)

      self.update()
   
   
   def __call__ (self, weights: ndarray) -> None:
      self.update(weights)
   
  
   def update(self, weights: Optional[ndarray|None] = None) -> None:
      
      self.__vals = self.__hidden_layer.get_values()

      if weights == None:   
         self.__output_val = np.exp(np.log(self.__vals)@self.__weights)
      else:
         self.__output_val = np.exp(np.log(self.__vals)@ weights)

      self.write_properties()

   
   def value(self) -> ndarray:
      return self.__output_val
   

   def write_properties(self):
      self.properties = {"input_vals": self.__vals, "weights": self.__weights, "output_val": self.__output_val}


   