from dataclasses import dataclass
import numpy as np
from numpy import random, sqrt
from heapq import merge
from sortedcontainers import SortedList

class Clock:
	t = 0
	dt = 0.3
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


def dn_dt(memory_size, synapticTC, a_eta0, phi):
	n = []
	n_ = []
	a_eta = a_eta0 # here a_eta0 is used as positive, and this function returns a positve curve
	eta = 0
	for _ in range(memory_size):
		dn_dt = a_eta*phi - eta/synapticTC 
		eta-= Clock.dt*dn_dt
		n.append(eta)
		n_.append(dn_dt)
	return np.array(n), np.array(n_)


class System:
	lr = 0.0001
	alpha = 0
	batch_size = 4
	decay = 0.618
	time_scale = 0
	bias_scale = 2
	membraneTC = 3.0
	synapticTC = 0.7
	k_eta = 4.0 
	k_epsilon = 2.0 
	beta = (membraneTC/synapticTC)
	a_eta0 = beta**(beta/(beta-1))/(k_eta*(beta-1))
	phi = (membraneTC - synapticTC)/(membraneTC * synapticTC)
	k_epsilon *=  phi
	k_eta *= phi
	membraneScale = 1-(Clock.dt/membraneTC)
	synapticScale = 1-(Clock.dt/synapticTC)
	trendScale = membraneScale - synapticScale
	memory_size = 100
	eta, dEta_dt = dn_dt(memory_size, synapticTC, a_eta0, phi)
	critical_threshold = 0.075

class Spike():
	def __init__(self, time, life = None, var = None):
		self.time = time
		self.life = life
		self.var = var
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
		return self.time<other.time
	def __le__(self, other):
		return self.time<=other.time
	def __gt__(self, other):
		return self.time>other.time
	def __ge__(self, other):
		return self.time>=other.time
	def __eq__(self, other):
		return self.time==other.time
	def __ne__(self, other):
		return self.time!=other.time
	def __mul__(self, factor):
		try:
			return Spike(self.time, self.time, self.var)
		except(e):
			raise(TypeError('Error in spike __mul__, product to spike must cast to float. \nException raised during __mul__'))
			print(e)
	def __rmul__(self, product):
		return self.__mul__(product)
	def __imul__(self, product):
		return self.__mul__(product)
	def __repr__(self):
		return "t + "+str(self.time)
	def __str__(self):
		return "t + "+str(self.time)
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
	def __init__(self, spikes=None, rate = 0):
		self.spikes = []
		if spikes:
			self.spikes = sorted(spikes)
		self.s_index = 0
		self.rate = 0
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
		self.bias = 0.1*np.random.ranf(self.num_units)
		self.threshold = np.ones(self.num_units)
		self.synapses = []
		self.spike_life = 0
		self.verbose = False
		self.destination = None
	def __len__(self):
		return len(self.num_units)
	def reset(self):
		self.epsilon = np.zeros(self.num_units)
		self.eta = np.zeros(self.num_units)
		self.a_epsilon = np.zeros(self.num_units)
		self.a_eta = np.zeros(self.num_units)
		self.t_s = np.zeros(self.num_units) - Clock.inf
		self.state = np.zeros(self.num_units)
		self.spikes = [Bus() for _ in range(self.num_units)]

	def connect(self, destination):
		self.destination= destination
		self.synapses = [sorted([Synapse(i,0.1+ 0.3*np.random.ranf(),10*np.random.ranf()) for i in range(destination.num_units)]) for _ in range(self.num_units)]
		self.spike_life+=destination.num_units
	def run(self, train = False):
		
		self.a_epsilon *= System.membraneScale
		self.a_eta *= System.membraneScale
		dEta_dt = -self.eta/System.synapticTC - self.a_eta*System.k_eta # remember, a_eta is stored positive but eta is negative,
		dEps_dt = -self.epsilon/System.synapticTC + (self.a_epsilon + self.bias)*System.k_epsilon
		self.eta += dEta_dt*Clock.dt
		self.epsilon += dEps_dt*Clock.dt

		self.state = np.where(self.epsilon + self.eta > self.threshold,True,False)
		spikes = self.state.nonzero()[0]

		dEps_dt+=dEta_dt
		if len(spikes):
			self.epsilon[spikes] = 0
			self.a_epsilon[spikes] = 0
			self.a_eta[spikes] = System.a_eta0*self.threshold[spikes]
			self.eta[spikes] = 0
		for s in spikes:
			var = {'dE_dt':dEps_dt[s], 'l':0, 'dEta_dt':dEta_dt[s], 'r':0} if train else None
			if self.synapses:
				self.spikes[s]+=Spike(Clock.t, self.spike_life, var)
			else:
				self.spikes[s]+=Spike(Clock.t, 0, var)

		return spikes

	def route(self, train= False):
		for bus, synapses in zip(self.spikes,self.synapses):
			l = len(synapses)
			for i, spike in enumerate(bus.alive()):
				spikeBreak = True
				dest_index = l-spike.life
				for d, synapse in enumerate(synapses[dest_index:]):
					if Clock.t-spike.time>=synapse.delay:
						spikeBreak = False
						self.destination.a_epsilon[synapse.address]+=synapse.weight
						spike.life-=1
						if self.verbose:
							print(f'{Clock.t}:{synapse.delay}', end = ' ')
					else:
						break
				if spikeBreak:
					break


class Synapse:
	def __init__(self, address, weight = 1, delay = 0):
		self.address = address
		self.weight = weight
		self.delay = delay

	def __lt__(self, other):
		return self.delay<other.delay
	def __le__(self, other):
		return self.delay<=other.delay
	def __gt__(self, other):
		return self.delay>other.delay
	def __ge__(self, other):
		return self.delay>=other.delay
	def __eq__(self, other):
		return self.delay==other.delay
	def __ne__(self, other):
		return self.delay!=other.delay

class Network:
	def __init__(self, layers):
		self.layers = [n if isinstance(n, Layer) else Layer(n) for n in layers]
		for i, n in enumerate(self.layers[:-1]):
			n.connect(self.layers[i+1])

	def run(self, input_impulses, num_iterations = None, train = False):
		for iteration in range(num_iterations if num_iterations else 1):
			self.layers[0].a_epsilon+=input_impulses
			for n in self.layers:
				n.run(train)
				n.route(train)
			Clock.tick()
	def reset(self):
		for l in self.layers:
			l.reset()

class UpdateGraph:
	def __init__(self, layer):
		self.layer = layer
		self.dB = np.zeros_like(layer.bias)
		if self.layer.destination:
			self.dW = np.zeros((layer.num_units, layer.destination.num_units)) 
			self.dD = np.zeros((layer.num_units, layer.destination.num_units))
		else:
			self.dW, self.dD = None, None
	def grad(self, freqs = None):
		# cosine cost for phase coding
		# time since first spike is considered  for coding
		# negative freq means neuron is not expected to fire
		# hence freqs = [1,2, -1,-2] means n1,n2 are rewarded with freqs 1 & 2 and n3,n4 are penalised
		rate = np.zeros(self.layer.num_units)
		if freqs is not None:
			signs = np.sign(freqs)
		hidden = self.layer.destination.num_units if self.layer.destination else None
		for n, bus in enumerate(self.layer.spikes):
			if bus.spikes:

				spike_index = len(bus.spikes) - 1
				if freqs is not None:
					start = bus.spikes[0].time
					bus.spikes[-1].var['l'] = np.sin((bus.spikes[-1].time - start)/freqs[n])*signs[n] # first spike delta doesnt depend on cost gradient because dC_dt will always be zero!
				for ii, spike in enumerate(reversed(bus.spikes[:-1])): # TODO check if critical
					if spike.var['dE_dt']>System.critical_threshold:

						if freqs is not None: # does this neuron have a cost associated

							prev_var = bus.spikes[spike_index - ii].var
							spike.var['l'] -= (np.sin((spike.time - start)/freqs[n])*signs[n] - prev_var['l']*prev_var['dEta_dt'])/spike.var['dE_dt']

						else:
							prev_var = bus.spikes[spike_index - ii].var
							spike.var['l'] += prev_var['l']*prev_var['dEta_dt']/spike.var['dE_dt']
					else:
						if freqs is not None:
							total_spikes = len(bus.spikes)
							if signs[n]>1:
								total_spikes=sqrt(1/total_spikes)
							spike.var['l']=10/sqrt(freqs[n]*signs[n]*(total_spikes))*signs[n]
					if hidden:
						bus.spikes[spike_index - ii].var['dEta_dt'] = np.zeros((2,hidden))
				if hidden:
					bus.spikes[0].var['dEta_dt'] = np.zeros((2,hidden))
				if self.dW is not None:
					for k, synapse in enumerate(self.layer.synapses[n]):
						delay = synapse.delay
						dest_bus = self.layer.destination.spikes[synapse.address]
						dRate = dest_bus.rate*synapse.weight/sqrt(self.layer.destination.num_units)
						bus.rate+=dRate
						self.dB[n]+=dRate*sqrt(len(bus.spikes))
						dest_index = len(dest_bus) - 1
						s_i = spike_index
						while(s_i>=0 and dest_index>=0):

							spike = bus.spikes[s_i]
							dest_spike = dest_bus.spikes[dest_index]
							dest_time = spike.time + delay
							diff = dest_spike.time - dest_time
							if diff>=0:
								if dest_index>0 and dest_bus.spikes[dest_index-1].time<dest_time:
									
									grad_index = int(diff/Clock.dt)
									if grad_index<System.memory_size:
										grad = System.dEta_dt[grad_index]/System.a_eta0 
										dE_dw = System.eta[grad_index]
									else:
										grad, dE_dw =0, 0
									# print(grad)
									spike.var['dEta_dt'][0][k] = dE_dw
									spike.var['dEta_dt'][1][k] = grad
									if spike.var['dE_dt'] > System.critical_threshold:
										spike.var['l'] -= grad*synapse.weight*dest_bus.spikes[dest_index].var['l']/spike.var['dE_dt'] # additive kernel serves as memory of contribution to activation 
									else:
										spike.var['l'] -= grad*synapse.weight*dest_bus.spikes[dest_index].var['l'] # additive kernel serves as memory of contribution to activation 
									s_i-=1
								dest_index-=1
							else:
								s_i-=1
					for k, synapse in enumerate(self.layer.synapses[n]):
						delay = synapse.delay
						dest_bus = self.layer.destination.spikes[synapse.address]
						for spike in bus.spikes:
							self.dW[n][k]+=spike.var['l']*spike.var['dEta_dt'][0][k]
							self.dD[n][k]+=spike.var['l']*spike.var['dEta_dt'][1][k]
							
					for spike in bus.spikes:
						del spike.var['dEta_dt']
						self.dB[n]+=spike.var['l']*System.k_eta*System.synapticTC
				else:
					for ii, spike in enumerate(bus.spikes): 
						self.dB[n]+=spike.var['l']*System.k_eta*System.synapticTC
			else:
				#no spikes
				if freqs is not None:
					dRate = 10/freqs[n]
					bus.rate += dRate
					self.dB[n]+=dRate
				else:
					for k, synapse in enumerate(self.layer.synapses[n]):
						dest_bus = self.layer.destination.spikes[synapse.address]
						dRate = dest_bus.rate*synapse.weight/sqrt(self.layer.destination.num_units)
						bus.rate += dRate
						self.dB[n]+=dRate
						self.dW[n][k]+=dRate

	def reset(self):
		self.dB = np.zeros_like(self.layer.bias)
		if self.layer.destination:
			self.dW = np.zeros((self.layer.num_units, self.layer.destination.num_units)) 
			self.dD = np.zeros((self.layer.num_units, self.layer.destination.num_units))
		else:
			self.dW, self.dD = None, None

	def update(self, lr, update_bias=True):
		if update_bias:
			self.layer.bias+=self.dB*lr
		if self.dW is not None:
			self.dW*=lr
			self.dD*=lr
			for i, synapses in enumerate(self.layer.synapses):
				for j, s in enumerate(synapses):
					s.weight+=self.dW[i][j] 
					s.delay+= self.dD[i][j] 
				self.layer.synapses[i] = sorted(self.layer.synapses[i])
		self.reset()

class Optimizer:
	def __init__(self, net, lr = 0.0001):
		self.lr = lr
		self.net = net
		self.nodes = []
		for n in net.layers:
			self.nodes.append(UpdateGraph(n))

	def grad(self, freqs):
		self.nodes[-1].grad(freqs)
		for n in reversed(self.nodes[:-1]):
			n.grad()

	def update(self):
		for i, n in enumerate(self.nodes):
			n.update(self.lr, update_bias=(i==len(self.nodes)-1))
	def reset(self):
		for n in self.nodes:
			n.reset()			


