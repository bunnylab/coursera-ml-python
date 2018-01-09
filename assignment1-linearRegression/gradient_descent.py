#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
	mo = (X * theta)
	htheta = mo[:,0] + mo[:,1]
	return (np.sum( (htheta - y)**2 )) / (2*len(y))
	

def gradientDescent(X, y, theta, alpha, num_iters):
	cost_h = []
	m = len(y)
	for i in range(num_iters):
		mo = (X * theta)
		htheta = mo[:,0] + mo[:,1]
		t0 = theta[0] - ( alpha * np.sum( (htheta - y)*X[:,0] ) / m )
		t1 = theta[1] - ( alpha * np.sum( (htheta - y)*X[:,1] ) / m )
		theta[0] = t0
		theta[1] = t1
		print( computeCost(X,y,theta) )
		
		
def plotFit(x, y, theta):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x, y)
	ax.plot(x, theta[1]*x + theta[0], '-', color='r')
	ax.set_xlabel('Population of City in 10,000s',fontsize=10)
	ax.set_ylabel('Profit in $10,000s',fontsize=10)
	plt.show()
	
	
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
	theta = np.zeros(2)
	alpha = np.float32(0.01)

	gradientDescent(ft_data[:,0:2], ft_data[:,2], theta, alpha, 1000)
	plotFit(ft_data[:,1], ft_data[:,2], theta)