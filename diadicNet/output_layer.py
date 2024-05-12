from numpy         import ndarray
from hidden_layer  import HiddenLayer
from output_neuron import OutputNeuron

class OtputLayer:
    
    def __init__(self,  hidden_layer: HiddenLayer) -> None:
        
        hidden_layer_vals    = hidden_layer.get_values()
        self.__output_neuron = OutputNeuron(hidden_layer_vals, hidden_layer)
        self.__value         = self.__output_neuron.value()

    
    def __call__(self, weigts: ndarray ) -> None:
        self.__output_neuron(weigts)
        self.__value = self.__output_neuron.value() 

    
    def value(self) -> float:
        return self.__value
    
    
    def properties(self) -> dict:
        return self.__output_neuron.properties

