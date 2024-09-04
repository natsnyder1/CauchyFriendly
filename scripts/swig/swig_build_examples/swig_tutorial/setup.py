# File : setup.py 
  
from distutils.core import setup, Extension 
#name of module 
name  = "gfg"
  
#version of module 
version = "1.0"
  
# specify the name of the extension and source files 
# required to compile this 
ext_modules = Extension(name='_gfg',sources=["gfg.i","gfg.c"]) 
  
setup(name=name, 
      version=version, 
      ext_modules=[ext_modules]) 