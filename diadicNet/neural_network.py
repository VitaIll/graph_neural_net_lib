from typing       import Optional

from numpy        import ndarray, array

from input_layer  import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OtputLayer

class DiadicNetwork:
    
    def __init__(
            
            self,
            input_ids:           list[tuple[int]],
            feature_list:        list[str],
            network_partition:   dict[tuple[int], ndarray[int]],
            factor_data:         ndarray[float],
            input_to_hidden_map: dict[tuple[int], list[tuple[int]]]
            
            ) -> None:
        
        self.input_ids    = input_ids

        self.input_layer  = InputLayer(neuron_ids = input_ids, feature_list= feature_list, network_partition = network_partition)
        self.hidden_layer = HiddenLayer(input_to_hidden_map = input_to_hidden_map, activation_function = 'expit', input_layer = self.input_layer)
        self.output_layer = OtputLayer(hidden_layer = self.hidden_layer, factor_data = factor_data)
        
        self.neuron_dim        = (len(feature_list)+1)
        self.hidden_weight_dim = (len(feature_list)+1) * len(input_ids)
        self.output_weight_dim = len(factor_data)
        self.weight_dim        = self.hidden_weight_dim + self.output_weight_dim
    
    
    def __call__( self, weights: dict ) -> float:
        
        weights_hidden = weights['hidden']
        weights_output = weights['output']

        self.hidden_layer(weights_hidden)
        self.output_layer(weights_output)

        return self.output_layer.value

