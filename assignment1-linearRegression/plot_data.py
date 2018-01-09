#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

tmp = []
with open('data/ex1data1.txt') as data:
	for line in data:
		tmp.append( [float(x) for x in line.split(',')] )

ft_data = np.array(tmp)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(ft_data[:,0], ft_data[:,1])
ax.set_xlabel('Population of City in 10,000s',fontsize=10)
ax.set_ylabel('Profit in $10,000s',fontsize=10)

plt.show()
