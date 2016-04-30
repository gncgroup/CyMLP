from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("xor", ["xor.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
 

setup(
    ext_modules=cythonize("xor.pyx"),
    include_dirs=[numpy.get_include()]
)    