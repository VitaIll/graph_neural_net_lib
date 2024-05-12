from numpy import ndarray
from typing import Optional

from input_layer  import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OtputLayer

class SimpleNetwork:
    
    def __init__(self, unit_id: int, unit_count: int, network_partition: list) -> None:
        
        self.__input_layer   = InputLayer(unit_id, network_partition)
        self.__hidden_layer  = HiddenLayer(unit_id, unit_count, "sigmoid", self.__input_layer)
        self.__output_layer  = OtputLayer(self.__hidden_layer)

    
    def __call__(self, hidden_params: ndarray, otput_params: Optional[ndarray] = None) -> None:
        self.__hidden_layer(hidden_params)

        if otput_params == None:
            self.__output_layer()
        else:
            self.__output_layer(otput_params)

    
    def __getitem__(self, layer_name: str) -> InputLayer|HiddenLayer|OtputLayer:
        match layer_name:
            
            case "input":
                return self.__input_layer
            
            case "hidden":
                return self.__hidden_layer
            
            case "output":
                return self.__output_layer
