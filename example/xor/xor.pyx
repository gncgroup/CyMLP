import neuronet  
cimport neuronet  


#Simple xor test
def train_and_test():   
	cdef neuronet.MLP nn
	layers_count=2
	neurons_count=[2,1]
	input_size=2
	nn=neuronet.MLP(input_size,neurons_count,layers_count)
	
	nn.read_weights("weights","prefix_")
	#Learning rate
	nn.set_n(1)    
	for i in range(2**10):
		nn.train([0.,0.],[0.])
		nn.train([0.,1.],[1.]) 
		nn.train([1.,1.],[0.]) 
		nn.train([1.,0.],[1.]) 
	print("Test:")
	print(str(nn.ask([0.,0.])[0]))
	print(str(nn.ask([0.,1.])[0]))
	print(str(nn.ask([1.,0.])[0]))
	print(str(nn.ask([1.,1.])[0]))
	nn.write_weights("weights","prefix_") 
	
	return
	
def run():
	train_and_test() 