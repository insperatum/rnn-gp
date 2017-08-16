#!conda/bin/python

#Visualise:
# Average population complexity	
# Novelty
# Output the most novel solutions, etc.


# Read selectivity pressures from the de jong textbook

# Todo:
#	- Multiple random seeds per program
#	- Reward high logq for winners
#	- Check/fix reward in tournament
#	- Single mutation rate for z
#   - Display first success for each program
#   - Batch Renormalization
#   - Program-dependent 'mutation/inference' network
#   - 'Commenting out'
#   - Programs calling programs
#   - QD
#   - change program
#   - stop being context free?
#   - fix resample_solved




import sys
import os
sys.path.append(os.getcwd()) #for slurm


import autograd.numpy as np
from autograd import grad
from autograd.core import getval
from autograd.optimizers import (sgd,adam)
from autograd.util import flatten
import math
import util
import data
import argparse
from mpi4py import MPI
from copy import deepcopy
import time
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from itertools import cycle
import pickle
from numpy.random import RandomState

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="test/" + datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))
parser.add_argument('--nTargets', type=int, default=50)
parser.add_argument('--nAgentsPerCore', type=int, default=1)
parser.add_argument('--layers', type=int, default=0)
parser.add_argument('--embeddingSize', type=int, default=50)
parser.add_argument('--sgd', type=str, default="adam")
parser.add_argument('--fitness', type=str, default="solved")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--novelty', type=float, default=0.)
parser.add_argument('--sgdIter', type=int, default=0)
parser.add_argument('--evoIter', type=int, default=9999999)
parser.add_argument('--sgdEvals', type=int, default=1)
parser.add_argument('--evoEvals', type=int, default=10)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--meta', dest='meta', action='store_true')
parser.add_argument('--deterministic', dest='deterministic', action='store_true')
parser.add_argument('--normalise', dest='normalise', action='store_true')
parser.add_argument('--variational', dest='variational', action='store_true')
parser.add_argument('--sgdPrint', type=int, default=10)
parser.add_argument('--evoPrint', type=int, default=1)
parser.add_argument('--saveEvery', type=int, default=100)
parser.add_argument('--selection', type=str, default="tournament()")
parser.add_argument('--initial_softmax', type=float, default=10) #5

parser.set_defaults(normalise=False)
parser.set_defaults(variational=False)
parser.set_defaults(verbose=False)
parser.set_defaults(meta=False)
parser.set_defaults(deterministic=False)

opt = parser.parse_args()
	
comm = MPI.COMM_WORLD
nAgents = comm.size*opt.nAgentsPerCore
isRoot = comm.rank == 0
if isRoot:
	print(opt)
	print("-"*50)
	print("Initialising %d agents on %d cores:" % (nAgents, comm.size))
	print("-"*50)
comm.Barrier()	

programs = data.createPrograms(opt.nTargets, seed=0)

actions = [
	# {'name':'const', 'terminal':True},

	{'name':'+',  'nargs':2, 'f':lambda args:lambda x,theta:args[0](x, theta) + args[1](x, theta)},
	# {'name':'call'},
	# {'name':'*',  'f':lambda args:lambda x:args[0](x) * args[1](x)},
	{'name':'sin','nargs':1, 'f':lambda args:lambda x,theta:np.sin(args[0](x, theta))},
	{'name':'cos','nargs':1, 'f':lambda args:lambda x,theta:np.cos(args[0](x, theta))},
	{'name':'-',  'nargs':1, 'f':lambda args:lambda x,theta:-args[0](x, theta)},
	{'name':'1',  'nargs':0, 'f':lambda args:lambda x,theta:1*np.ones(*x.shape)},
	{'name':'2',  'nargs':0, 'f':lambda args:lambda x,theta:2*np.ones(*x.shape)},
	{'name':'3',  'nargs':0, 'f':lambda args:lambda x,theta:3*np.ones(*x.shape)},
	{'name':'x',  'nargs':0, 'f':lambda args:lambda x,theta:x},
	{'name':'θ',  'nargs':0, 'f':lambda args:lambda x,theta:theta},
	# {'name':'tau', 'terminal':True,  'f':lambda args:lambda x:x*0+np.pi/2},
	# {'name':'cos',  'f':lambda args:lambda x:np.cos(args[0](x))}
]
maxDepth=2



if isRoot:
	programsAtDepth = []
	programsAtDepth.append(len([action for action in actions if action['nargs']==0]))
	for i in range(1,maxDepth+1):
		programsAtDepth.append(0)
		for action in actions:
			if action['nargs']>0:
				programsAtDepth[i] += np.power(programsAtDepth[i-1], action['nargs'])
	print("Possible programs at each depth:", programsAtDepth, flush=True)
	print("Total:", np.sum(programsAtDepth), flush=True)
	numPossiblePrograms = np.sum(programsAtDepth)


def squareParams():
	return {
		# 'weight':np.eye(opt.embeddingSize) + np.random.randn(opt.embeddingSize)*0.01,
		'weight':{
			'val':np.random.randn(opt.embeddingSize) / np.sqrt(opt.embeddingSize),
			'mutation_sd':0.1 / np.sqrt(opt.embeddingSize)},
		'bias':{
			'val':np.random.randn(opt.embeddingSize) * 0.01,
			'mutation_sd':0.1}
		}
def linear(theta, z):
	return np.dot(theta['weight']['val'], z) + theta['bias']['val']

def newAgent():
	genome = {
		# 'z':[{'val':np.random.randn(opt.embeddingSize), 'resampleIfUnsolved':{'idx':i}, 'mutation_sd':0.1, 'pCrossover':0.5} for i in range(opt.nTargets)],
		# 'z_logvar':[{'val':np.ones(opt.embeddingSize)*-1, 'mutation_sd':0.1, 'pCrossover':0.5} for i in range(opt.nTargets)]
		
		'log_lr':{'val':np.log(opt.lr), 'mutation_sd':0.1},
		'decoder':{
			'val':{
				'h':[squareParams() for i in range(opt.layers)],
				'softmax_weight':{'val':np.random.randn(len(actions), opt.embeddingSize)  * opt.initial_softmax / np.sqrt(opt.embeddingSize), 'mutation_sd':0.1},
				'const':{
					'weight':{'val':np.random.randn(1, opt.embeddingSize) / np.sqrt(opt.embeddingSize), 'mutation_sd':0.1/np.sqrt(opt.embeddingSize)},
					'bias':  {'val':np.random.randn(opt.embeddingSize) * 0.01, 'mutation_sd':0.1}},
				# 'call':{
				# 	'weight':{'val':np.random.randn(opt.nTargets, opt.embeddingSize) / np.sqrt(opt.embeddingSize), 'mutation_sd':0.1/np.sqrt(opt.embeddingSize)},
				# 	'bias':  {'val':np.random.randn(opt.nTargets) * 0.01, 'mutation_sd':0.1}},
			},
			'pCrossover':0.5
		}
	}
	for action in actions:
		if 'nargs' in action:
			genome['decoder']['val'][action['name']] = [squareParams() for _ in range(action['nargs'])]

	return {'genome':genome}

def getParamsByName(genome, paramname):
	def recurse(x, prefix = None):
		if type(x) is dict:
			params = {k:v for k0 in x for k,v in recurse(x[k0], (prefix + "." if prefix is not None else "") + k0).items()}
			if paramname in x:
				params[prefix] = x[paramname]
			return params
		elif type(x) is list and len(x)>0:
			if len(x)>1:
				idxs = [0, len(x)-1]
			else:
				idxs = [0]
			# params = recurse(x[0], prefix+"[0]")
			params = {k:v for i in idxs for k,v in recurse(x[i], prefix + "[" + str(i) + "]").items()}
			return params
		else:
			return {}
	return recurse(genome)

def initialiseZ(agents):
	for agent in agents:
		agent['genome']['z']={
			'val':[{'val':
				{'seed': {'val':np.random.randint(np.iinfo(np.int32).max, size=opt.evoEvals), 'resampleIfUnsolved':{'idx':i, 'p':0.2, 'type':'integer'}},
				'mu': {'val':np.random.randn(opt.embeddingSize), 'resampleIfUnsolved':{'idx':i, 'p':0.2, 'type':'gaussian'}, 'mutation_sd':0.1},
				'logvar': {'val':np.ones(opt.embeddingSize)*-1, 'resetIfUnsolved':{'idx':i, 'val':np.ones(opt.embeddingSize)*-1}, 'mutation_sd':0.2}},
				'pCrossover':0.5
			} for i in range(opt.nTargets)]}

		if opt.normalise: agent['genome']['z']['normalise'] = "mu"
		agent['genome']['baselines'] = [0. for _ in programs]


def decode(genome, i, k): #i = program index, k = eval number
	decoder = genome['decoder']
	def decodeInner(z, prng, depth=0):
		h = z
		for i in range(opt.layers):
			h = util.elu(linear(decoder['val']['h'][i], h))
		logprobs = util.logsoftmax(np.dot(decoder['val']['softmax_weight']['val'], h))
		
		if depth>=maxDepth:
			possible_idxs = np.array([i for i in range(len(actions)) if actions[i]=="const" or ("nargs" in actions[i] and actions[i]["nargs"]==0)])
			probs = np.exp(logprobs)[possible_idxs]
			probs = probs / sum(probs)
			if opt.deterministic:
				action_idx = possible_idxs[np.argmax(getval(probs))]
				logq=0
			else:
				action_idx_idx = prng.choice(range(len(possible_idxs)), p=getval(probs))
				action_idx = possible_idxs[action_idx_idx]
				logq=np.log(probs[action_idx_idx])
		else:
			probs = getval(np.exp(logprobs))
			if opt.deterministic:
				action_idx = np.argmax(probs)
				logq=0
			else:
				action_idx = prng.choice(np.arange(len(logprobs)), p=probs)
				logq = logprobs[action_idx]
		action = actions[action_idx]
			
		if action['name']=='const':
			c = np.mean(linear(decoder['val']['const'], h))
			f = lambda x:np.ones(x.shape) * c
			name="%.2f" % getval(c)
		# elif action['name']=='call':
		# 	logprobs = util.logsoftmax(linear(decoder['val']['call'], h))
		# 	probs = getval(np.exp(logprobs))
		# 	probs = probs / sum(probs)
		# 	program_idx = np.random.choice(np.arange(opt.nTargets), p=probs)
		# 	d = decodeInner(genome['z'][program_idx]['val'], depth+1)
		# 	logq = logq + logprobs[program_idx] + d['logq'] 
		# 	name = "[" + "%2d"%program_idx + "]"
		# 	f = lambda x: d['f'](x)
		# 	return {'f':f, 'logq':logq, 'name':name}
		else:
			zs = [linear(params, h) for params in decoder['val'][action['name']]]
			ds = [decodeInner(z, prng, depth+1) for z in zs]
			args = [d['f'] for d in ds]
			logqs = [d['logq'] for d in ds]
			logqs = [d['logq'] for d in ds]
			f = action['f'](args)
			if len(logqs)>0:logq = logq + np.sum(np.array(logqs))
			if len(args)==0:
				name = action['name']
			else:
				name = action['name'] + "(" + ",".join([d['name'] for d in ds]) + ")"
		return {'f':f, 'logq':logq, 'name':name}

	z_mu = genome['z']['val'][i]['val']['mu']['val']
	seed = genome['z']['val'][i]['val']['seed']['val'][k]
	prng = RandomState(seed)
	if opt.variational:
		z_logvar = genome['z']['val'][i]['val']['logvar']['val']
		z = z_mu + prng.randn(*z_mu.shape)*np.sqrt(np.exp(z_logvar))
		kl = -0.5 * (1 + z_logvar - np.exp(z_logvar) - z_mu*z_mu)
	else:
		z = z_mu
	d = decodeInner(z, prng)
	d['y'] = d['f'](data.x, data.theta)
	if opt.variational: d["kl"] = kl
	return d
	# prediction = z[0]*x*x + z[1]*x + z[2]
	

def programLoss(d, prog):
	diff = d['y'] - prog['y']
	meanabsdiff = np.mean(np.abs(diff))
	solved = meanabsdiff < 0.01

	if opt.fitness == "solved":
		loss = 0 if solved else math.inf
	elif opt.fitness == "sqdiff":
		loss = np.mean(np.square(diff))
	elif opt.fitness == "absdiff":
		loss = meanabsdiff

	if opt.variational:
		loss = loss + np.sum(d["kl"])

	return loss, solved

# def surrogateLoss(genome, iter=0):
# 	def progSurrogateLoss(i):
# 		def singleRun():
# 			d = decode(genome, i)
# 			loss, solved = programLoss(d, programs[i])
# 			baseline = genome['baselines'][i]
# 			return loss + kl + (getval(loss) - getval(baseline)) * (d['logq']) + 100*np.square(getval(loss) - baseline)
# 		return np.mean(np.array([singleRun() for _ in range(opt.sgdEvals)]))
# 	return np.mean(np.array([progSurrogateLoss(i) for i in range(len(programs))]))
# loss_grad = grad(surrogateLoss)

def callback(genome, iter, gradient):
	if isRoot:
		if iter % opt.sgdPrint == 0:
			print(getSummary({"genome":genome}, "(Gradient iteration " + str(iter) + "/" + str(opt.sgdIter) + ")", verbose=opt.verbose))

def getSummary(agent, name, stats=None, verbose=False):
	totalLoss = 0
	totalSolved = 0
	progList=""
	for i in range(len(programs)):
		d = decode(agent["genome"], i, 0)
		loss, solved = programLoss(d, programs[i])
		totalSolved = totalSolved + solved
		totalLoss = totalLoss + loss
		if verbose:
			progList = progList + "\n" + programs[i]["name"].rjust(20," ")
			progList = progList + (" ✔" if solved else "  ")
			progList = progList + "\t\t\t"
			if "novelty" in agent:
				progList + ("♫ " if agent["novelty"][i]>0.5 else "  ")
			progList = progList + d['name'].ljust(40," ")
			progList = progList + "\tBaseline:"+ "%2.3f" % agent["genome"]['baselines'][i]
			progList = progList + "\tLoss:" + "%2.3f" % loss
			# progList = progList + "\tKL:" + "%2.3f" % kl
	string = name + "\tLoss:" + str(totalLoss/opt.nTargets) + "\tSolved:" + str(totalSolved) + "/" + str(opt.nTargets)
	# string = name + "\tSolved:" + str(totalSolved) + "/" + str(opt.nTargets)

	# mutation_sd = {'z':[], 'decoder':[]}
	# resample_rates = {'z':[], 'decoder':[]}
	# def collectMutate(tag, x):
	# 	nonlocal mutation_sd
	# 	# nonlocal resample_rates
	# 	if type(x) is dict and 'mutate' in x:
	# 		mutation_sd[tag] += [x['mutate']]
		# if type(x) is dict and 'resample' in x:
		# 	resample_rates[tag] += [x['resample']]
	# util.recurse(lambda x: collectMutate("z", x), genome["z"])
	# util.recurse(lambda x: collectMutate("decoder", x), genome["decoder"])
	
	if opt.meta:
		string = string + "\n" + "Learning Rate:" + str(np.exp(agent["genome"]["log_lr"]['val']))
		# string = string + "\tz resampling:" + "%.3f"%np.min(resample_rates["z"]) + "-" + "%.3f"%np.max(resample_rates["z"])
		# string = string + "\tdecoder mutation:" + "%.3f"%np.min(mutation_sd["decoder"]) + "-" + "%.3f"%np.max(mutation_sd["decoder"]) + "}" + "\n"
	# if stats is not None:
	# 	string = string + "\n" + " ".join(["%s: %2.2f" % (k, stats[k]) for k in stats])
	if verbose: string = string + progList + "\n"
	return string


def tournament(alpha=1,sampleInner=False,select="multiple",proportion=0.5): # select in {"single","multiple"}
	if select=="single":
		nPlayers = int(nAgents*proportion)
	else:
		nPlayers = int(nAgents)

	def f(agents):
		def match():
			player_idxs = np.random.choice(range(len(agents)), nPlayers, replace=False)
			player_scores = np.zeros(len(player_idxs))
			for programIdx in range(opt.nTargets):
				losses = np.array([alpha*agents[i]["losses"][programIdx] - opt.novelty * agents[i]["novelty"][programIdx] for i in player_idxs])
				probs = np.exp(util.logsoftmax(-losses))
				if not np.any(np.isnan(probs)):
					if sampleInner:
						winner_idx = np.random.choice(range(len(player_idxs)), p=probs)
						player_scores[winner_idx] += 1
					else:
						player_scores += probs
			if select=="single":
				best_idx = player_idxs[np.argmax(player_scores)]
				return agents[best_idx]
			else:
				best_idxs = sorted(range(len(player_idxs)), key=lambda i:player_scores[i], reverse=True)
				return [agents[player_idxs[i]] for i in best_idxs[:int(nAgents*proportion)]]
		if select=="single":
			return [match() for _ in range(len(agents))]
		else:
			return match()
	return f

# def tournament(nChallenge=int(opt.nTargets/2), nPlayers=int(nAgents/2)):
# 	def f(agents):
# 		def match():
# 			score_idxs = np.random.choice(range(len(agents[0]["losses"])), nChallenge, replace=False)
# 			player_idxs = np.random.choice(range(len(agents)), nPlayers, replace=False)
# 			# best_idx = max(player_idxs, key=lambda i:np.sum(agents[i]["solved"][score_idxs]))
# 			best_idx = min(player_idxs, key=lambda i:np.sum(agents[i]["losses"][score_idxs] - opt.novelty * agents[i]["novelty"][score_idxs]))
# 			return agents[best_idx]
# 		return [match() for _ in range(len(agents))]
# 	return f

# def resample_solved():
# 	def f(agents):
# 		scores = np.array([np.sum(agent["solved"]) for agent in agents]) # This KL thing is just stupid...
# 		# scores = scores - (1/100)*np.array([np.sum(agent["kl"]) for agent in agents])
# 		# for j in range(nTargets):


# 		scores = scores + opt.novelty * np.array([np.sum(agent["novelty"]) for agent in agents])

# 		# 	if np.max([agent["solved"][j] for agent in agents])==0:
# 		# 		print("index " + str(j) + " unsolved!")
# 		# 		scores = scores + np.array([agent["novelty"] for agent in agents])
# 		probs = np.exp(util.logsoftmax(scores))
# 		winner_idxs = np.random.choice(range(len(agents)), len(agents), p=probs)
# 		winners = [agents[i] for i in winner_idxs]
# 		winners = sorted(winners, key=lambda agent:(-np.sum(agent["solved"])))

# 		numSolved = np.array([np.sum(agent["solved"]) for agent in winners])
# 		print("Winners solved: %.1f (±%.1f)" % (np.mean(numSolved), np.std(numSolved)))
# 		return winners
# 	return f


def offspring():
	def f(agents):
		def operate(xs):
			x = xs[0]
			if type(x) is dict:
				for k in x:
					operate([x[k] for x in xs])
				if 'pCrossover' in x and np.random.rand()<x['pCrossover']:
					if np.random.rand()<0.5: (xs[0]['val'], xs[1]['val']) = (xs[1]['val'], xs[0]['val'])
				if 'mutation_sd' in x:
					if opt.meta: x['mutation_sd'] *= np.exp(np.random.randn() * 0.1)
					x['val'] += np.random.randn(*x['val'].shape)*x['mutation_sd']
				if 'resampleIfUnsolved' in x:
					progIdx = int(x['resampleIfUnsolved']['idx'])
					if np.random.rand() > agents[0]["solved"][progIdx] and np.random.rand()<x['resampleIfUnsolved']['p']:
						if x['resampleIfUnsolved']['type'] == "gaussian":
							x['val'] = np.random.randn(*x['val'].shape)
						elif x['resampleIfUnsolved']['type'] == "integer":
						 	x['val'] = np.random.randint(np.iinfo(np.int32).max, size=x['val'].shape)
						else:
							raise ValueError("Unknown resampleIfUnsolved type")
				if 'resetIfUnsolved' in x:
					progIdx = int(x['resetIfUnsolved']['idx'])
					if np.random.rand() > agents[0]["solved"][progIdx]:
						x['val'] = x['resetIfUnsolved']['val']
				if 'normalise' in x:
					key = x['normalise']
					for xi in xs:
						meanval = np.mean([xij["val"][key]['val'] for xij in xi["val"]], 0)
						stdval = np.std([xij["val"][key]['val'] for xij in xi["val"]], 0)
						for xij in xi["val"]:
							xij['val'][key]['val'] -= meanval
							xij['val'][key]['val'] /= stdval
			elif type(xs[0]) is list:
				for i in range(len(xs[0])): 
					operate([x[i] for x in xs])

		def newChild():
			xs = [deepcopy(agents[i]["genome"]) for i in np.random.choice(range(len(agents)), 2)]
			operate(xs)
			return {"genome": xs[0]}
		return [newChild() for _ in range(nAgents)]
	return f


# params_scatter, unflatten_scatter = flatten(coreAgents)
# paramss_scatter = np.empty((comm.size, *params_scatter.shape))

if isRoot:
	print("\n" + "-"*50)
	print("Training")
	print("-"*50)


def getStats(agents, winners, champion, t, nUniquePrograms, nUniqueFunctions):
	stats = {}
	agentSolved = [np.sum(agent["solved"]) for agent in agents]
	winnersSolved = [np.sum(agent["solved"]) for agent in winners]
	stats["solved_avg"] = np.mean(agentSolved)
	stats["solved_winners"] = np.mean(winnersSolved)
	stats["solved_champion"] = np.sum(champion["solved"])
	stats["time"] = t
	stats["unique_winners"] = len(np.unique([id(agent) for agent in winners]))
	stats["unique_winners_pred"] = len(agents) * (1 - np.power((len(agents)-1)/len(agents), len(agents)))
	stats["unique_programs"] = nUniquePrograms
	stats["unique_functions"] = nUniqueFunctions
	stats["entropy_avg"] = -np.mean([agent["logq"] for agent in agents])
	stats["entropy_winners"] = -np.mean([agent["logq"] for agent in winners])
	stats["entropy_champion"] = -champion["logq"]
	stats["mutation_stats"] = {}
	for k in agents[0]["mutation_stats"]:
		stats["mutation_stats"][k] = np.mean([agent["mutation_stats"][k] for agent in agents])
	return stats

def saveFigures(name, prefix, epochHistory):
	def makeFig(ylabel, keys, legend, filename, yzero=False, getStats=lambda stats:stats, legendSize=None, cycleStyle=False, logscale=False, ymax=None):
		fig = plt.figure(figsize=(6, 6))
		ax1 = fig.add_subplot(111)
		times = [stats["time"] for stats in epochHistory]
		takeEvery = max(1, int(len(epochHistory)/50))
		
		lines = ["-","--","-.",":"] if cycleStyle else ["-"]
		linecycler = cycle(lines)
		for key in keys:
			ax1.plot(times[::takeEvery], [getStats(stats)[key] for stats in epochHistory[::takeEvery]], next(linecycler))
	
		ax1.set_ylabel(ylabel)
		ax1.set_xlim(xmin=0)
		if yzero: ax1.set_ylim(ymin=0)
		if logscale: ax1.semilogy()
		if ymax is not None: ax1.set_ylim(ymax=ymax)
		ax2 = ax1.twiny()
		ax2.set_xlim(ax1.get_xlim())

		ax1.xaxis.tick_top()
		ax1.xaxis.set_label_position('top')
		ax2.xaxis.tick_bottom()
		ax2.xaxis.set_label_position('bottom')

		if times[-1]>60*60*6:
			ax1.set_xticks(range(0,int(times[-1]),60*60*2))
			ax1.set_xticklabels([time.strftime("%-H hours", time.gmtime(t)) for t in range(0,int(times[-1]),60*60*2)])
		elif times[-1]>60*60:
			ax1.set_xticks(range(0,int(times[-1]),60*60))
			ax1.set_xticklabels([time.strftime("%-H hours", time.gmtime(t)) for t in range(0,int(times[-1]),60*60)])
		elif times[-1]>60*30:
			ax1.set_xticks(range(0,int(times[-1]),60*10))
			ax1.set_xticklabels([time.strftime("%-M mins", time.gmtime(t)) for t in range(0,int(times[-1]),60*10)])
		elif times[-1]>60*5:
			ax1.set_xticks(range(0,int(times[-1]),60*5))
			ax1.set_xticklabels([time.strftime("%-M mins", time.gmtime(t)) for t in range(0,int(times[-1]),60*5)])
		else:
			ax1.set_xticks(range(0,int(times[-1]),60))
			ax1.set_xticklabels([time.strftime("%-M mins", time.gmtime(t)) for t in range(0,int(times[-1]),60)])

		interval = max(1, int(util.roundDown(len(epochHistory)/5)))
		ax2.set_xticks([epochHistory[i]["time"] for i in range(interval-1, len(epochHistory), interval)])
		ax2.set_xticklabels(range(interval, len(epochHistory)+1, interval))
		ax2.set_xlabel("Generation")

		plt.suptitle(name,fontsize=14)
		plt.title("(%d individuals on %d cores)" % (nAgents, comm.size), y=1.08, fontsize=11)
		plt.gcf().subplots_adjust(bottom=0.3   , top=0.85)
		
		fig.text(0.02, 0.2, ", ".join([k + "=" + str(getattr(opt, k)) for k in vars(opt)]),
			wrap=True, horizontalalignment='left', verticalalignment='top', fontsize=9,
			bbox=dict(facecolor='none' ))
		
		if legendSize is not None:
			ax1.legend(legend, loc='upper left', fontsize=legendSize)
		else:
			ax1.legend(legend, loc='upper left')

		plt.savefig(filename) 		
		plt.close()

	makeFig("# Tasks Solved", ['solved_champion', 'solved_winners', 'solved_avg'],['Champion', 'Winners', 'Population'], "experiments/" + prefix + ".png", yzero=True)
	makeFig("# Selected Individuals", ['unique_winners', 'unique_winners_pred'],['Unique Winners', 'Random Selection'], "experiments/" + prefix + "_winners.png", yzero=True)
	makeFig("Diversity", ['unique_programs', 'unique_functions'],['Unique Programs', 'Unique Functions'], "experiments/" + prefix + "_diversity.png", yzero=True, ymax=numPossiblePrograms)
	makeFig("Conditional Entropy", ['entropy_champion', 'entropy_winners', 'entropy_avg'], ['Champion', 'Winners', 'Population'], "experiments/" + prefix + "_entropy.png", yzero=True)
	makeFig("Mutation s.d.", [k for k in epochHistory[0]["mutation_stats"]], [k for k in epochHistory[0]["mutation_stats"]], "experiments/" + prefix + "_mutation.png", logscale=True, getStats=lambda stats: stats["mutation_stats"], legendSize=6, cycleStyle=True)



def loadCheckpoint():
	try:
		state = pickle.load(open("experiments/" + opt.name + "_checkpoint.p", "rb"))
		if isRoot: print("Loading checkpoint at epoch %d generation %d" % (state["epoch"], state["generation"]), flush=True)
		return state
	except OSError as e:
	    state =  {"history":[],
	    		"epoch":0, "generation":0,
	    		"agents":[newAgent() for _ in range(nAgents)]}
	    return state
	
def saveCheckpoint(state):
	if isRoot:
		print("Saving checkpoint at epoch %d generation %d" % (state["epoch"], state["generation"]))
		pickle.dump(state, open("experiments/" + opt.name + "_checkpoint.p", "wb"))

state = loadCheckpoint()

while True:
	if state["generation"]==0:
		state["epoch"] = state["epoch"]+1
		state["time"] = 0
		state["history"].append([])
		programs = data.createPrograms(opt.nTargets, seed=state["epoch"])
		initialiseZ(state["agents"])
		if isRoot: print("Epoch: %d\n" % state["epoch"])

	for state["generation"] in range(state["generation"]+1,opt.evoIter):
		lastTime = time.time()
		times = []
		def timePoint(name):
			global lastTime
			times.append({"name":name, "t":time.time()-lastTime})
			lastTime = time.time()

		coreAgentss = util.list_chunk(state["agents"], opt.nAgentsPerCore) if isRoot else None
		comm.Barrier()
		timePoint("wait")
		# comm.Scatter(paramss_scatter, params_scatter, root=0)	
		coreAgents = comm.scatter(coreAgentss, root=0)
		timePoint("scatter")



		for agent in coreAgents:
			if opt.sgdIter>0:
				agent["genome"] = eval(opt.sgd)(loss_grad, agent["genome"], step_size=np.exp(agent["genome"]['log_lr']['val']), num_iters=opt.sgdIter,
					callback=callback if isRoot and agent==coreAgents[0] else None)
			decodedProgs = [[decode(agent["genome"], j, k) for k in range(opt.evoEvals)] for j in range(opt.nTargets)]
			# def solved(j):
			# 	return np.max([np.mean([programSolved(prog, programs[j]) for prog in progs]) for progs in decodedProgs])
			# agent["solved"] = [solved(j) for j in range(opt.nTargets)]
			programLosses = [[programLoss(d, programs[j]) for d in decodedProgs[j]] for j in range(opt.nTargets)] #(loss, isSolved)
			agent["losses"] = np.array([np.mean([l[0] for l in losses]) for losses in programLosses])
			agent["solved"] = np.array([np.mean([l[1] for l in losses]) for losses in programLosses])
			agent["programnames"] = [[d['name'] for d in progs] for progs in decodedProgs]
			agent["functionhashes"] = [[util.hasharray(d['y']) for d in progs] for progs in decodedProgs]
			agent["logq"] = np.mean([d['logq'] for progs in decodedProgs for d in progs])
			agent["mutation_stats"] = getParamsByName(agent["genome"], "mutation_sd")
			# print(agent["programnames"])
		timePoint("evaluate")
		# params_gather, unflatten_gather = flatten(coreAgents)
		# paramss_gather = np.empty((comm.size, *params_gather.shape)) if isRoot else None
		# timePoint("flatten")
		comm.Barrier()
		timePoint("wait")
		# comm.Gather(params_gather, paramss_gather, root=0)
		coreAgentss = comm.gather(coreAgents, root=0)
		timePoint("gather")
		if isRoot:
			# coreAgentss = [unflatten_gather(params) for params in paramss_gather]
			# timePoint("unflatten")
			state["agents"] = [agent for coreAgents in coreAgentss for agent in coreAgents]


			all_program_names= [programname for agent in state["agents"] for programnames in agent['programnames'] for programname in programnames]
			program_name_counter = Counter(all_program_names)
			all_function_hashes = [functionhash for agent in state["agents"] for functionhashes in agent['functionhashes'] for functionhash in functionhashes]
			function_hash_counter = Counter(all_function_hashes)

			for agent in state["agents"]:
				agent["novelty"] = np.array([np.mean([1/program_name_counter[programname] for programname in programnames]) for programnames in agent["programnames"]])
			timePoint("aggregate")
			winners = eval(opt.selection)(state["agents"])
			timePoint("select")
			champion = state["agents"][np.argmax([np.sum(agent["solved"]) for agent in state["agents"]])]
			stats = getStats(state["agents"], winners, champion, state["time"] + np.sum([t["t"] for t in times]), nUniquePrograms=len(program_name_counter), nUniqueFunctions=len(function_hash_counter))
			state["history"][-1].append(stats)
			# fig, ax = plt.subplots( nrows=1, ncols=1 ) 
			if state["generation"]%opt.evoPrint==0:
				print("\n" + getSummary(champion, "Epoch %d Generation %d" %(state["epoch"], state["generation"]), stats=stats, verbose=True))

			timePoint("stats")

			state["agents"] = offspring()(winners)
			timePoint("offspring")

			if state["generation"]%opt.saveEvery==0:
				saveFigures(
					name = opt.name + "   Epoch %d"%state["epoch"],
					prefix = opt.name + "_epoch_%d" % state["epoch"],
					epochHistory = state["history"][-1])
				saveCheckpoint(state)
				timePoint("save")
			
			
			# paramss_scatter[:] = [flatten(coreAgents)[0] for coreAgents in coreAgentss]
			# timePoint("flatten")
		# coreAgents = unflatten_scatter(params_scatter)
		# timePoint("unflatten")
		if isRoot:
			print("Timings:   " + ", ".join(["%s=%.2fs"%(t["name"], t["t"]) for t in times]) + " | Total: %.2fs"%np.sum([t["t"] for t in times]) + "\n" + "-"*50, flush=True)
		state["time"] += np.sum([t["t"] for t in times])
	state["generation"] = 0