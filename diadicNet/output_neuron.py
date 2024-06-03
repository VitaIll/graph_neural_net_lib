from numpy         import ndarray, append, array
from hidden_neuron import HiddenNeuron

class OutputNeuron:

   def __init__(self, neuron_list: list[HiddenNeuron], factor_data: ndarray[float]) -> None:
      
      self.source_neurons =  neuron_list
      self.factors        =  factor_data
      self.output_val     =  0
   
   
   def __call__ (self, weights: ndarray[float]) -> None:
      
      val_list = []

      for neuron in self.source_neurons:
         val_list.append(neuron.output_value)
   
      w               = array(val_list)
      self.output_val = self.cb_function(weights, w)
   
   
   def cb_function(self, weights:ndarray[float], w: ndarray[float]) -> float:

      multiplier = weights[0]
      coefs      = weights[1:]
      last_coef  = 1 - coefs.sum()
      coefs      = append(coefs,last_coef)
      coefs      = coefs/coefs.sum()

      output     = multiplier

      for i in range(len(self.factors)):
         output *= (self.factors[i] * w[i]) ** coefs[i]

      return output
   