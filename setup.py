from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[Extension("neuronet", ["neuronet.c"],include_dirs=[numpy.get_include()],extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ], extra_link_args=['-fopenmp'])],
)
 

setup(
    ext_modules=cythonize("neuronet.pyx"),
    include_dirs=[numpy.get_include()]
)    
