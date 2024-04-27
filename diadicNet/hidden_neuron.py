
import numpy as np

from scipy.special import logit
from diadicNet.input_neuron import InputNeuron

class HiddenNeuron:
   
   def __init__(self, input_neuron_list: list, weigts: np.array) -> None:
      
      self.features = np.array([])

      for input_neuron in input_neuron_list:
         
         if isinstance(input_neuron, InputNeuron):

            self.features = np.append(self.features, input_neuron.m_feature_extractor.m_feature_list)
            self.weigte
         else:
            raise TypeError
      
      self.w = HiddenNeuron.activation_function(self.features)
    
   def activation_function (features: np.array) -> float:
      return logit(features)
      