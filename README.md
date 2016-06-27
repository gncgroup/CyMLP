# CyMLP
High-performance parallel Cython/C implementation of multilayer perceptron.

To compile use:
python setup.py build_ext --inplace

To set the number of threads use:
export OMP_NUM_THREADS=8