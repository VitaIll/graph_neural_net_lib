
import numpy as np

from scipy.special import expit
from typing        import Optional

class HiddenNeuron:
   
   def __init__(
                self,
                idx_pair:            tuple, 
                activation_function: Optional[str] = "sigmoid"
                ) -> None:
   
      self.features                   = np.array([1])
      self.activation_function_label  = activation_function
      self.idx_pair                   = idx_pair

   def __call__ (self, input_neuron_list: list,  weights: np.array) -> None:
           
           self.weights               = weights

           for input_neuron in input_neuron_list:
              self.features = np.append(self.features, input_neuron.m_feature_extractor.m_feature_list)
            
           self.weighted_value = self.weights @ self.features
           self.output_value   = self.activation_function()
    
   #--------------------------------------------------------------------------------------------------#
   
   def activation_function (self) -> float:
      ''' Activation function of the hidden layer neuron.'''

      match self.activation_function_label:
        
         case  "sigmoid":
            return expit(self.weighted_value)
      