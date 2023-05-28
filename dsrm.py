from dataclasses import dataclass, field
import numpy as np
from numpy import random, sqrt
from heapq import merge
from sortedcontainers import SortedList
class Clock:
	t = 0
	dt = 0.02
	inf = 100
	@staticmethod
	def tick():
		Clock.t+=Clock.dt
	@staticmethod
	def run(dt):
		Clock.t+=dt
	@staticmethod
	def set_dt(dt):
		Clock.dt=dt
		System.calibrate()

class System:
	lr = 0.0001
	alpha = 0
	batch_size = 4
	decay = 0.618
	synapse_scale = 4
	time_scale = 0
	bias_scale = 2
	membraneTC = 3.0
	synapticTC = 0.7
	k_epsilon = 2.0
	k_eta = 4.0
	
	beta = (membraneTC/synapticTC)
	a_eta0 = beta**(beta/(beta-1))/(k_eta*(beta-1))
	phi = (membraneTC - synapticTC)/(membraneTC * synapticTC)
	membraneScale = 1-(Clock.dt/membraneTC)
	synapticScale = 1-(Clock.dt/synapticTC)
	trendScale = membraneScale - synapticScale
	batch_num = 0

	@staticmethod
	def calibrate():       # needs to be called whenever system parameters mTC, sTC, K or Clock.dt change
		System.beta = (membraneTC/synapticTC)
		System.a_eta0 = System.beta**(System.beta/(System.beta-1))/(System.k_eta*(System.beta-1))
		System.phi = (System.membraneTC - System.synapticTC)/(System.membraneTC * System.synapticTC)
		System.membraneScale = 1-(Clock.dt/System.membraneTC)
		System.synapticScale = 1-(Clock.dt/System.synapticTC)
		System.trendScale = System.membraneScale - System.synapticScale

class Spike():
	def __init__(self, time, weight = 1, life = None, options = None):
		self.time = time
		self.weight = weight
		self.life = life
		self.options = options
	def copy(self, life = None, options = None):
		return Spike(self.time, self.weight, life, options)
	# def inform(self, destination, external_weight = 1):
	# 	destination.a_epsilon+= (self.weight * external_weight)
	# 	self.life-=1
	def __add__(self, adduct):
		if isinstance(adduct, Bus):
			return adduct.__add__(self)
		else:
			try:
				return float(adduct)+self.time
			except Exception as e:
				raise(TypeError('adduct to spike must cast to float'))
	def __radd__(self, adduct):
		return self.__add__(adduct)
	def __iadd__(self, adduct):
		self.time = self.__add__(adduct)
		return self
	def __sub__(self, adduct):
		try:
			return self.time - float(adduct)
		except Exception as e:
			raise(TypeError('adduct to spike must cast to float'))
	def __rsub__(self, adduct):
		try:
			return float(adduct) - self.time
		except Exception as e:
			raise(TypeError('adduct to spike must cast to float'))
	def __isub__(self, adduct):
		self.time = self.__sub__(adduct)
		return self
	def __lt__(self, other):
		if(isinstance(other,Spike)):
			return self.time<other.time
		else:
			return self.time<other
	def __le__(self, other):
		if(isinstance(other,Spike)):
			return self.time<=other.time
		else:
			return self.time<=other
	def __gt__(self, other):
		if(isinstance(other,Spike)):
			return self.time>other.time
		else:
			return self.time>other
	def __ge__(self, other):
		if(isinstance(other,Spike)):
			return self.time>=other.time
		else:
			return self.time>=other
	def __eq__(self, other):
		if(isinstance(other,Spike)):
			return self.time==other.time
		else:
			return self.time==other
	def __ne__(self, other):
		if(isinstance(other,Spike)):
			return self.time!=other.time
		else:
			return self.time!=other
	def __mul__(self, factor):
		try:
			return Spike(self.time, float(factor)*self.weight, self.time, self.options)
		except(e):
			raise(TypeError('product to spike must cast to float'))
	def __rmul__(self, product):
		return self.__mul__(product)
	def __imul__(self, product):
		return self.__mul__(product)
	def __repr__(self):
		return "t + "+str(self.time)+" * "+str(self.weight)
	def __str__(self):
		return "t + "+str(self.time)+" * "+str(self.weight)
	def __bool__(self):
		if self.life is not None:
			return bool(self.life)
		else:
			return True
	def __float__(self):
		return self.time
	def __hash__(self):
		return hash(self.time)

class Bus():
	def __init__(self, spikes=None):
		self.spikes = []
		if spikes:
			self.spikes = sorted(spikes)
		self.s_index = 0
	def alive(self):
		for i, s in enumerate(self.spikes[self.s_index:]):
			if s.life>0:
				return self.spikes[i:]
			else:
				self.s_index+=1
		return []
	def dead(self):
		for i, s in enumerate(self.spikes[self.s_index:]):
			if s.life>0:
				return self.spikes[:i]
			else:
				self.s_index+=1
		return self.spikes
	def split(self, low, high = None):
		newBus = Bus()
		for i, s in enumerate(self.spikes):
			if s>low:
				if not high:
					newBus.spikes = self.spikes[i:]
				else:
					for j, t in enumerate(self.spikes[i:]):
						if t>high:
							newBus.spikes = self.spikes[i:j]
							return newBus
					newBus.spikes = self.spikes[i:]
		return newBus
	def copy(self, life = None):
		newBus = Bus()
		if life == 'copy':
			newBus.spikes = [s.copy(s.life) for s in self.spikes]
		elif life is not None:
			newBus.spikes = [s.copy(life) for s in self.spikes]
		else:
			newBus.spikes = [s.copy() for s in self.spikes]
	def __bool__(self):
		return bool(self.spikes)
	def __len__(self):
		return len(self.spikes)
	def __add__(self, newSpikes):
		finalSpikes = None
		if isinstance(newSpikes, list):
			finalSpikes = list(merge(self.spikes,sorted(newSpikes)))
		elif isinstance(newSpikes, Bus):
			finalSpikes = list(merge(self.spikes,newSpikes.spikes))
		elif isinstance(newSpikes, Spike):
			finalSpikes = list(merge(self.spikes, [newSpikes]))
		else:
			raise(TypeError('Adduct to bus must be Spike, pure list of Spikes or Bus. Don\'t make lists of Buses and Spikes, just cumulate a Bus for optimal merging.'))
		finalBus = Bus()
		finalBus.spikes = finalSpikes
		return finalBus
	def __radd__(self, adduct):
		return self.__add__(adduct)
	def __iadd__(self, adduct):
		return self.__add__(adduct)
	def __str__(self):
		return "Bus with "+str(len(self.spikes))+" spikes"
	def __repr__(self):
		s = "BUS("
		l = len(self.spikes)
		if l:
			if l==1:
				return s + repr(self.spikes[0]) + ")"
			else:
				s+=repr(self.spikes[0])
				for sp in self.spikes[1:]:
					s+=', '+repr(sp)
				s+=")"
				return s
	def __getitem__(self, i):
		return self.spikes[i]

	def __iter__(self):
		return iter(self.spikes) # you'll see what I mean

class Layer():
	def __init__(self, num_units):
		self.num_units = num_units
		self.epsilon = np.zeros(self.num_units)
		self.eta = np.zeros(self.num_units)
		self.a_epsilon = np.zeros(self.num_units)
		self.a_eta = np.zeros(self.num_units)
		self.t_s = np.zeros(self.num_units) - Clock.inf
		self.state = np.zeros(self.num_units)
		self.spikes = [Bus() for _ in range(self.num_units)]
		self.bias = np.zeros(self.num_units)
		self.threshold = np.ones(self.num_units)
		self.synapses = {} # every destination layer has a set of synapses
		self.destinations = {}
		self.expected = None
		self.spike_life = 0
		self.verbose = False

	def reset(self):
		self.epsilon = np.zeros(self.num_units)
		self.eta = np.zeros(self.num_units)
		self.a_epsilon = np.zeros(self.num_units)
		self.a_eta = np.zeros(self.num_units)
		self.t_s = np.zeros(self.num_units) - Clock.inf
		self.state = np.zeros(self.num_units)
		self.spikes = [Bus() for _ in range(self.num_units)]
		self.expected = None

	def connect(self, destination):
		new_key = str(len(self.destinations.keys()))
		self.destinations[new_key] = destination
		self.synapses[new_key] = [sorted([Synapse(i,np.random.ranf(),0) for i in range(destination.num_units)]) for _ in range(self.num_units)]
		self.spike_life+=destination.num_units
	def run(self, learn = False):
		outs = []
		self.a_epsilon *= System.synapticScale
		self.a_eta *= System.synapticScale
		self.eta = self.eta*System.membraneScale - self.a_eta*System.trendScale*System.k_eta
		self.epsilon = self.epsilon*System.membraneScale + (self.a_epsilon + self.bias)*System.trendScale*System.k_epsilon
		self.state = np.where(self.epsilon + self.eta > self.threshold,True,False)
		spikes = self.state.nonzero()[0]
		if len(spikes):
			self.epsilon[spikes] = 0
			self.a_epsilon[spikes] = 0
			self.a_eta[spikes] = System.a_eta0*self.threshold[spikes]
			self.eta[spikes] = 0
		for s in spikes:
			if self.synapses:
				self.spikes[s]+=Spike(Clock.t, 1, self.spike_life, None)
			else:
				self.spikes[s]+=Spike(Clock.t, 1, 0, None)
		if learn and self.expected:
			self.state_learn = np.zeros(self.num_units)
			for i in range(len(self.expected)):
				if Clock.t >=self.expected[i][0]:
					self.expected = self.expected[1:]
					self.state_learn[i]==1
		return spikes
	def route(self, learn = False):
		for destination_key in self.destinations.keys():
			destination_layer = self.destinations[destination_key] 
			for bus, synapses in zip(self.spikes,self.synapses[destination_key]):
				l = len(synapses)
				for spike in bus.alive():
					spikeBreak = True
					addresses = []
					for synapse in synapses[l-spike.life:]:
						if Clock.t-spike.time>=synapse.delay:
							spikeBreak = False
							# spike.inform(,synapse.weight)
							destination_layer.a_epsilon[synapse.address]+=synapse.weight*spike.weight
							spike.life-=1
							if self.verbose:
								print(f'{Clock.t}:{synapse.delay}', end = ' ')
						else:
							break
					if spikeBreak:
						break

	def delE_delTi(self, t_i, w_i, t = None):
		''' Partial derivative of EPSP w.r.t. timestamp of an input spike through a given synapse. Eta isn't directly dependent so this is only partial derivative of Epsilon. '''
		if not t:
			t = Clock.t
			if t_i > self.t_s:
				g = System.k_epsilon*(exp((t_i - t)/System.membraneTC)/System.membraneTC - exp((t_i - t)/System.synapticTC)/System.synapticTC)*w_i
			else:
				g = 0
		elif self.spikes.spikes:		
			rs = list(reversed([-2*Clock.inf] +[s.time for s in self.spikes.spikes] + [self.spikes.spikes[-1].time + Clock.inf]))
			g = 0
			for s, s_ in zip(rs[:-1], rs[1:]):
				if t <= s and t>s_ and t_i <= s and t_i>s_:
					g = System.k_epsilon*(exp((t_i - t)/System.membraneTC)/System.membraneTC - exp((t_i - t)/System.synapticTC)/System.synapticTC)*w_i
					break
		else:
			g = System.k_epsilon*(exp((t_i - t)/System.membraneTC)/System.membraneTC - exp((t_i - t)/System.synapticTC)/System.synapticTC)*w_i
		
		return g

	def delE_delTi(self, t_i, w_i, t = None, indices = None, invert_indices = True):
		''' Partial derivative of EPSP w.r.t. timestamp of an input spike through a given synapse. Eta isn't directly dependent so this is only partial derivative of Epsilon. '''
		t_s = self.t_s
		if indices:
			t_s = t_s[indices]
		if not t:
			t = Clock.t
			g = np.where(t_i>t_s, System.k_epsilon*(exp((t_i - t)/System.membraneTC)/System.membraneTC - exp((t_i - t)/System.synapticTC)/System.synapticTC)*w_i, 0)
		else:
			g = []
			for i, t_j in enumerate(t):
				spikes = self.spikes[i].spikes
				g.append(0 if np.any(np.where(spikes > t_i[i] and spikes < t_j,True, False)) else System.k_epsilon*(exp((t_i - t)/System.membraneTC)/System.membraneTC - exp((t_i - t)/System.synapticTC)/System.synapticTC)*w_i)
			g = np.array(g)
		if indices is not None and invert_indices:
			output = np.zeros(self.num_units)
			output[indices] = g
			return output
		return g

	def dEta_dT(self):
		''' Derivative of Eta w.r.t. time.'''
		return -self.eta/System.synapticTC - System.k_eta*self.a_eta*System.phi # a_eta is stored positive but theoretically it is a negative current

	def delE_delTs(self):
		''' (Partial) derivative of EPSP w.r.t. timestamp of latest spike. Previous timestamps dont matter due to neglected trailing currents.'''

		# remember, a_eta is stored positive but theoretically it is a negative current

		g = [bool(s) for s in self.spikes]
		return np.where(g, self.eta/System.synapticTC + System.k_eta*System.phi*self.a_eta- System.k_epsilon*System.phi*self.biasCurrent*(self.a_eta/(self.threshold*System.a_eta0))**(1/System.beta) ,0)

	def dE_dBias(self):
		''' Partial derivative of EPSP w.r.t. bias current. Eta isn't dependent so this is only partial derivative of Epsilon. '''
		return System.k_epsilon*System.phi*(1-(self.a_eta/(self.threshold*System.a_eta0))**(1/System.beta))*System.membraneTC

class Synapse:
	def __init__(self, address, weight = 1, delay = 0):
		self.address = address
		self.weight = weight
		self.delay = delay
		self.dW = 0
		self.dD = 0
		self.training_variables = {}

	def learn(self):
		pass
	def reset(self):
		self.training_variables = {}
		
	def __lt__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay<other.delay
		else:
			try:
				return self.delay <other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')
	def __le__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay<=other.delay
		else:
			try:
				return self.delay <=other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')
	def __gt__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay>other.delay
		else:
			try:
				return self.delay>other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')
	def __ge__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay>=other.delay
		else:
			try:
				return self.delay>=other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')
	def __eq__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay==other.delay
		else:
			try:
				return self.delay==other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')
	def __ne__(self, other):
		if(isinstance(other,Synapse)):
			return self.delay!=other.delay
		else:
			try:
				return self.delay!= other
			except:
				raise TypeError('Synapse only comparable to float or Synapse')

class RBM:
	def __init__(self, nodes):
		self.nodes = [n if isinstance(n, Layer) else Layer(n) for n in nodes]
		for i, n in enumerate(self.nodes[:-1]):
			n.connect(self.nodes[i+1])

	def run(self, input_impulses, num_iterations = None, learn = False):
		for iteration in range(num_iterations if num_iterations else 1):
			self.nodes[0].a_epsilon+=input_impulses
			for n in self.nodes:
				n.run()
				n.route()



			

