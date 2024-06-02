
import urllib

from dataclasses  import dataclass, field
from urllib.parse import urlparse


class Utils:

    @staticmethod
    def function_exists(module, function_name: str) -> bool:
        '''
           Check for existance of a function in a third party library,
           module arg is the tartget library.

        '''
        return hasattr(module,function_name) and callable(getattr(module, function_name))
    
    

@dataclass
class Reference:
    '''
       Stores references to documentation for used third-party libraries in case error is raised. 
    '''
    networkx_link: urllib.parse.ParseResult = field(default=urlparse("https://networkx.org/documentation/stable/reference/algorithms/index.html"), init=False)
