from input_neuron import InputNeuron

class InputLayer:
    ''' 
        Input layer of the neural network.
          - for now assumes output for a single macro unit, e.g. one firm/ state etc..
          - network partition is assumed to be repressented by list of sub_matricies,
            we can also consider alternative containers.
          - neuron list updated to dictionary
    '''
    
    def __init__(self, modeled_unit: int, network_partition: list) -> None:
           
           self.neuron_list = {}
           self.count       = 1

           for sub_matrix in network_partition:
                 
                 idx_tuple       = (modeled_unit, self.count)
                 neuron          = InputNeuron (idx_tuple ,sub_matrix)
                 neuron_key_pair = {idx_tuple: neuron}
                 
                 self.neuron_list.update(neuron_key_pair)

                 self.count += 1

    
    def __getitem__(self, key: tuple) -> list:
          
          neuron  = self.neuron_list[key]

          return neuron.properties