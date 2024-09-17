from numpy          import ndarray, zeros
from typing         import Any, Optional
from neural_network import DiadicNetwork

class ErrorFunction:
    
    def __init__(
            
            self, 
            network: DiadicNetwork,
            target:  ndarray,
            perturb_param: Optional[float] = 0.1
            
            ) -> None:
         
         self.network    = network
         self.target     = target
         self.input_ids  = network.input_ids
         self.neuron_dim = network.neuron_dim
         self.perturb    = perturb_param
        
         pass
    
    def __call__(
            
             self, 
             weigths: ndarray,
            **kwds: Any    

            ) -> float:
        
        self.weights  = weigths
        
        weights_dict  = {}
   

        hidden_pars   = self.weights[:10]
        
        hidden_dict  = {
                            (1,2): hidden_pars[:4],
                            (1,1): hidden_pars[4:6],
                            (1,3): hidden_pars[6:]
           }
        
        weights_dict['hidden'] = hidden_dict
        weights_dict['output'] = self.weights[10:]

        res = 0
        for tar in self.target:
            res += (self.network(weights_dict) - tar)**2
        
        res = res*(1/len(self.target))

        return res
    
    
    def grad(self, weigths: ndarray) -> ndarray:

        g = zeros(len(weigths))
        h = zeros(len(weigths))

        for i in range(len(weigths)):
            h[i]        += self.perturb
            change      = (self.__call__(weigths + h) - self.__call__(weigths))/self.perturb
            g[i]        = change

        return g
  

class ErrorFunctionV2:
    
    def __init__(
            
            self, 
            network: DiadicNetwork,
            train_data,
            perturb_param: Optional[float] = 0.1
            
            ) -> None:
         
         self.network    = network
         self.train_data = train_data
         self.input_ids  = network.input_ids
         self.neuron_dim = network.neuron_dim
         self.perturb    = perturb_param
         self.record     = []
    
    def __call__(
            
             self, 
             weigths: ndarray,
            **kwds: Any    

            ) -> float:
        
        self.weights  = weigths
        self.record.append(weigths)

        weights_dict  = {}
   

        hidden_pars   = self.weights[:5]
        
        hidden_dict  = {
                            (1,1): hidden_pars[:5]
                      
           }
        
        weights_dict['hidden'] = hidden_dict
        weights_dict['output'] = self.weights[5:]

        res = 0
        
        for adj_matrix, tar in self.train_data:
 
            self.network.input_layer({(1,1): adj_matrix})
            res += (self.network(weights_dict) - tar)**2
        
        res = res*(1/len(self.train_data))

        return res
    
    
    def grad(self, weigths: ndarray) -> ndarray:

        g = zeros(len(weigths))
        h = zeros(len(weigths))

        for i in range(len(weigths)):
            h[i]        += self.perturb
            change      = (self.__call__(weigths + h) - self.__call__(weigths))/self.perturb
           # print((self.__call__(weigths + h),self.__call__(weigths) ))
            g[i]        = change

        return g