from dsrm import *
import matplotlib.pyplot as plt
net = Network([100,10,1])

def excite(ll, peak_voltage = 0.1):
	for i in range(ll.num_units):	
		ll.a_epsilon[i]+=np.random.ranf()*peak_voltage

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
for i in range(100):
	# excite(net.layers[0])
	net.run(np.random.ranf(100)*0.1)
	for ii, j in enumerate(net.layers[0].state):
		if j:
			x1.append(i)
			y1.append(ii)
	for ii, j in enumerate(net.layers[1].state):
		if j:
			x2.append(i)
			y2.append(ii)
	x3.append(i)
	y3.append(net.layers[0].eta[0] + net.layers[0].epsilon[0])
	Clock.tick()
ax1.scatter(x1, y1, s=2)
ax2.scatter(x2, y2, s=2)
ax3.plot(x3, y3)
fig.show()
plt.show()