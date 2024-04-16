from setuptools import setup, find_packages

setup (
    
    name         = 'diadicNet',
    version      =  '1.0.0',
    packages     =  find_packages(),  
    license      = 'MIT',
    author       = 'Vit Illichmann',
    author_email = 'vit.Illichmann@gmail.com',

    install_requires = [  'numpy', 'networkx'  ]
 )
