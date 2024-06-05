from numpy         import ndarray
from hidden_layer  import HiddenLayer
from output_neuron import OutputNeuron

class OtputLayer:
    
    def __init__(self,  hidden_layer: HiddenLayer, factor_data: ndarray[float]) -> None:
        
        self.source_neurons = hidden_layer.neuron_list
        self.factor_data    = factor_data
        self.output_neuron  = OutputNeuron(self.source_neurons, self.factor_data)
    

    
    def __call__(self, weigts: ndarray ) -> None:
        self.output_neuron(weigts)
        self.value = self.output_neuron.output_val

    


