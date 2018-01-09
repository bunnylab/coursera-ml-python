#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
	mo = (X * theta)
	htheta = mo[:,0] + mo[:,1]
	return (np.sum( (htheta - y)**2 )) / (2*len(y))
	
	
if __name__ == "__main__":
	tmp = []
	with open('data/ex1data1.txt') as data:
		for line in data:
			tmp.append( [float(x) for x in line.split(',')] )

	# create np array and add column of ones for theta0
	ft_data = np.array(tmp)
	theta0 = np.ones((len(ft_data),1))
	ft_data = np.append(theta0, ft_data, axis=1)

	# initialize starting fit parameters
	theta = np.zeros((1,2))
	alpha = np.float32(0.01)

	j0 = computeCost(ft_data[:,0:2], ft_data[:,2], theta)
	print(j0)