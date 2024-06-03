from scipy         import special
from numpy         import array, ndarray
from typing        import Optional
from utils         import Utils
from input_neuron  import InputNeuron 


class HiddenNeuron:
   
   def __init__(
                self,
                id:                  tuple[int], 
                input_neurons:       list[InputNeuron],
                activation_function: Optional[str]     = "expit"
                ) -> None:
      
      ''' Initiate id, list of input neurons and activation function. '''
      
      self.id            = id
      self.input_neurons = input_neurons

      if Utils.function_exists(activation_function):
          self.activation_function = getattr(special, activation_function)
      else:
          raise ValueError(f"{activation_function} is not available in scipy.special module.")
        
      
      self.__get_features()
   


   def __call__ (self,  weights: ndarray[float]) -> None:
      ''' Compute transformed weighted value of features.'''
           
      self.weights        = weights
      weighted_value      = self.weights @ self.features
      self.output_value   = self.activation_function(weighted_value)

      self.properties     = {"neuron_id":self.id, "weigths": self.weights, "features": self.features,"output_value": self.output_value}
  

   
   def __get_features(self) -> None:
    '''Extract fetures from referenced instances of input neurons.'''
    
    feature_list = []
    
    for neuron in self.input_neurons:
          features  = [1.] + list(neuron.features.values())
          feature_list.append(features)
     
    self.features = array(feature_list)
   