
import numpy as np

from scipy.special import expit
from typing        import Optional

class HiddenNeuron:
   
   def __init__(
                self,
                idx_pair:            tuple, 
                feature_list:        list,
                activation_function: Optional[str] = "sigmoid"
                ) -> None:
      
      features      =  [1.] + feature_list
      self.features = np.array(features)

      self.activation_function_label  = activation_function
      self.idx_pair                   = idx_pair

      weights = np.random.random(size = len(self.features))
      self.__call__(weights= weights)


   def __call__ (self,  weights: np.array) -> None:
           
           self.weights        = weights
           self.weighted_value = self.weights @ self.features
           self.output_value   = self.activation_function()
         
   #--------------------------------------------------------------------------------------------------#
   
   def activation_function (self) -> float:
      ''' Activation function of the hidden layer neuron.'''

      match self.activation_function_label:
        
         case  "sigmoid":
            return expit(self.weighted_value)
      