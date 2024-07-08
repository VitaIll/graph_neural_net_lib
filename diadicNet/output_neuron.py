from numpy         import ndarray, append, array, exp, log
from hidden_neuron import HiddenNeuron

class OutputNeuron:

   def __init__(self, neuron_list: list[HiddenNeuron], factor_data: ndarray[float]) -> None:
      
      self.source_neurons =  neuron_list
      self.factors        =  log(factor_data)
      self.output_val     =  0
   
   
   def __call__ (self, weights: ndarray[float]) -> None:
      
      val_list = []

      for neuron in self.source_neurons.values():
         val_list.append(neuron.output_value)
   
      w               = array(val_list)
      self.output_val = self.cb_function(weights, w)
   
   
   def cb_function(self, weights:ndarray[float], w: ndarray[float]) -> float:
       
       x          = weights * w
       output     = x @ self.factors
      
       return exp(output)
   