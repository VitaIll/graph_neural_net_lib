from input_neuron import InputNeuron

class InputLayer:
    ''' 
        Input layer of the neural network.
          - for now assumes output for a single macro unit, e.g. one firm/ state etc..
          - network partition is assumed to be repressented by list of sub_matricies,
            we can also consider alternative containers.
    '''
    def __init__(self, modeled_unit: int, network_partition: list) -> None:
           
           self.neuron_list = []
           self.count       = 1

           for sub_matrix in network_partition:
                 
                 idx_tuple     = (modeled_unit, self.count)
                 neuron        = InputNeuron (idx_tuple ,sub_matrix)
                
                 self.neuron_list.append(neuron)
                 
                 self.count    += 1
                 
 