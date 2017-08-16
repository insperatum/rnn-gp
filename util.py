import autograd.numpy as np
from autograd.scipy.misc import logsumexp

def logsoftmax(x):
	return x - logsumexp(x)

def elu(x, alpha=1):
	return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))

def list_chunk(arr, n):
	return [arr[i:i + n] for i in range(0, len(arr), n)]

def roundDown(x):
	tens = np.power(10, np.floor(np.log10(x)))
	if x>=tens*5:
		return tens*5
	elif x>=tens*2:
		return tens*2
	else:
		return tens

def hasharray(numpyarray):
	return hash(tuple(map(float, numpyarray)))
	# numpyarray.flags.writeable=False
	# print(numpyarray.data, flush=True)
	# h = hash(numpyarray.data)
	# numpyarray.flags.writeable=True
	# return h