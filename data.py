import numpy as np

from numpy.random import RandomState
prng = RandomState(0)

x = prng.rand(50)*2-1
theta = prng.rand(50)*2-1

def makeConst(prng):
	c = prng.choice([1,2,3])
	return {'name':str(c), 'nargs':0, 'f':lambda x,theta,args: np.ones(x.shape)*c}
ops = [
	{'p': 0.2, 'type':'scalar', 'create':makeConst},
	{'p': 0.2, 'type':'scalar', 'create':lambda prng: {'name':"x",   'nargs':0, 'f':lambda x,theta,args: x}},
	{'p': 0.2, 'type':'angle',  'create':lambda prng: {'name':"θ",   'nargs':0, 'f':lambda x,theta,args: theta}},
	{'p': 0.1, 'type':'scalar', 'create':lambda prng: {'name':"+",   'childType':'scalar', 'nargs':2, 'f':lambda x,theta,args: args[0]+args[1]}},
	{'p': 0.1, 'type':'angle',  'create':lambda prng: {'name':"+",   'childType':'angle',  'nargs':2, 'f':lambda x,theta,args: args[0]+args[1]}},
	# {'p': 0.2,  'create':lambda: {'name':"*",   'nargs':2, 'f':lambda x,args: args[0]*args[1]}},
	{'p': 0.05, 'type':'angle', 'create':lambda prng: {'name':"-", 'childType':'angle', 'nargs':1, 'f':lambda x,theta,args: -args[0]}},
	{'p': 0.05, 'type':'scalar', 'create':lambda prng: {'name':"-", 'childType':'scalar', 'nargs':1, 'f':lambda x,theta,args: -args[0]}},

	{'p': 0.1, 'type':'scalar', 'create':lambda prng: {'name':"sin", 'childType':'angle', 'nargs':1, 'f':lambda x,theta,args: np.sin(args[0])}},
	{'p': 0.05, 'type':'scalar', 'create':lambda prng: {'name':"cos", 'childType':'angle',  'nargs':1, 'f':lambda x,theta,args: np.cos(args[0])}}
]
maxDepth=2

def createProgram(existingPrograms, prng):
	valid_ops = ops

	probs = [op['p'] for op in valid_ops]
	probs = probs / np.sum(probs)
	opClass = prng.choice(valid_ops, p=probs)
	op = opClass['create'](prng)
	def findExistingProgram(outputType):
		valid_progs = [prog for prog in existingPrograms if prog['type']==outputType]
		if len(valid_progs)==0: return None
		return prng.choice(valid_progs)
	children = [findExistingProgram(outputType=op['childType']) for _ in range(op['nargs'])]
	if None in children:
		return None
	depth = 0 if len(children)==0 else 1+np.max([child["depth"] for child in children])
	if depth>maxDepth: return None

	def f(x, theta):
		args = [child['f'](x, theta) for child in children]
		return op['f'](x, theta, args)
	if op['nargs']==0:
		name = op['name']
	else:
		name = op['name'] + '(' + ','.join([child['name'] for child in children]) + ')'

	return {'name':name, 'f':f, 'y':f(x, theta), 'type':opClass['type'], 'depth':depth}

def createPrograms(n, seed=None):
	prng = RandomState(seed)
	programs = []
	while len(programs)<n:
		newProg = createProgram(programs, prng)
		if newProg is None:
			continue
		sameProg = next((prog for prog in programs if all(np.abs(newProg['y']-prog['y'])<1e-7)), None)
		if sameProg:
			if len(newProg['name']) < len(sameProg['name']): sameProg.update(newProg)
		# elif 'x' in newProg['name']:
		else:
			programs.append(newProg)
	return sorted(programs, key=lambda prog: (len(prog['name']), prog['name']))


# def createProgram(depth=1, outputType=None):
# 	if depth>3:
# 		return None

# 	if outputType is None:
# 		valid_ops = ops
# 	else:
# 		valid_ops = [op for op in ops if op['type']==outputType]

# 	probs = [op['p'] for op in valid_ops]
# 	probs = probs / np.sum(probs)
# 	op = np.random.choice(valid_ops, p=probs)['create']()
# 	children = [createProgram(depth+1, outputType=op['childType']) for _ in range(op['nargs'])]
# 	if None in children:
# 		return None

# 	def f(x, theta):
# 		args = [child['f'](x, theta) for child in children]
# 		return op['f'](x, theta, args)
# 	if op['nargs']==0:
# 		name = op['name']
# 	else:
# 		name = op['name'] + '(' + ','.join([child['name'] for child in children]) + ')'

# 	return {'name':name, 'f':f, 'y':f(x, theta)}

# def createPrograms(n, seed=None):
# 	if seed is not None:
# 		np.random.seed(seed)
# 	programs = []
# 	while len(programs)<n:
# 		newProg = createProgram()
# 		if newProg is None:
# 			continue
# 		sameProg = next((prog for prog in programs if all(np.abs(newProg['y']-prog['y'])<1e-7)), None)
# 		if sameProg:
# 			if len(newProg['name']) < len(sameProg['name']): sameProg.update(newProg)
# 		# elif 'x' in newProg['name']:
# 		else:
# 			programs.append(newProg)
# 	return sorted(programs, key=lambda prog: (len(prog['name']), prog['name']))