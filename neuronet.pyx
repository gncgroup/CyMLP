cdef extern from "arrayinit.c":
	int* int_arr_init(int size)
	float* float_arr_init(int size)
	float** float_arr_init_2d(int size_x,int size_y) 
	float** float_arr_init_2d_p(int size_x,int* size_y)
	float*** float_arr_init_3d(int size_x,int* size_y,int* size_z)
cdef extern from "compute.c":	
	float* fprop_compute(float* vector,float*** weights,float** res,int* layers,int layers_count)
	float* backprop_compute(float* vector,float* correct_output,float n,float*** weights,float** res,int* layers,int layers_count)
from libc.stdlib cimport free
from libc.math cimport log	
from libc.math cimport exp
from libc.math cimport tanh
from libc.math cimport abs 
import numpy as np 
cimport numpy as np
cimport cython
import copy
import random  
import os.path 
cdef class MLP:
	def __init__(self,input_length,layers,layers_count):
		self.n=1
		self.layers_count=layers_count
		self.layers=int_arr_init(layers_count+1)
		self.layers[0]=input_length
		for i in range(layers_count):
			self.layers[i+1]=layers[i]
		self.res=float_arr_init_2d_p(self.layers_count+1,self.layers)
		self.init_weights()
	
	cdef float* ask(self,float* vector): 
		return fprop_compute(vector,self.weights,self.res,self.layers,self.layers_count)
	
	cdef float* train(self,float* vector,float* correct_output): 
		self.ask(vector) 
		return backprop_compute(vector,correct_output,self.n,self.weights,self.res,self.layers,self.layers_count)
	def init_weights(self):  
		f_weights=[]
		for l in range(self.layers_count):
			f_weights.append(np.random.rand(self.layers[l+1],self.layers[l]+1)-0.5)
				
		self.weights=float_arr_init_3d(self.layers_count,self.get_neurons_on_layers(),self.get_weights_size_on_layers())
		for l in range(self.layers_count):
			for n in range(self.layers[l+1]): 
				for w in range(self.layers[l]+1):
					self.weights[l][n][w]=copy.deepcopy(f_weights[l][n][w])

	def read_weights(self,path,prefix=""): 
		global weights
		save=False 
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w"
			if os.path.isfile(full_path): 
				f_weights = np.loadtxt(full_path,dtype = np.float32).reshape(self.layers[l+1],self.layers[l]+1) 
				for n in range(self.layers[l+1]):
					for w in range(self.layers[l]+1):    
						self.weights[l][n][w]=copy.deepcopy(f_weights[n,w])  
						
	def write_weights(self,path,prefix=""): 
		global weights 
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w" 
			f=open(full_path,'w')  
			for n in range(self.layers[l+1]):
				for w in range(self.layers[l]+1):  
					f.write(str(self.weights[l][n][w])+"\n")
			f.close()
					
	cdef void set_n(self,float n):
		self.n=n
					
	cdef int* get_neurons_on_layers(self):
		cdef int* neurons_on_layers
		neurons_on_layers=int_arr_init(self.layers_count)
		for i in range(self.layers_count):
			neurons_on_layers[i]=self.layers[i+1]
		return neurons_on_layers 
		
	cdef int* get_weights_size_on_layers(self):
		cdef int* neurons_on_layers
		neurons_on_layers=int_arr_init(self.layers_count)
		for i in range(self.layers_count):
			neurons_on_layers[i]=self.layers[i]+1
		return neurons_on_layers 		