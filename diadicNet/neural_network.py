from numpy        import ndarray

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
        
        self.input_layer = InputLayer(input_ids, feature_list)