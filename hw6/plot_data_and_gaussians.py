import numpy as np
import pylab as P
import matplotlib.mlab as mlab

from plot_gauss_parameters import *


def plot_data_and_gaussians(data_file_name, mu1, mu2, covar1, covar2):
    
    # load data
    data = np.genfromtxt(data_file_name, delimiter=' ')

    # plot data as a scatter plot
    P.scatter(data[:,0], data[:,1], s=20, c='k', marker='x', alpha=.65, linewidths=2)

    # plot gaussian #1
    plot_gauss_parameters(mu1, covar1, 'r')

    # plot gaussian #2
    plot_gauss_parameters(mu2, covar2, 'b')

    P.show()

if __name__ == "__main__":
    data = np.genfromtxt("dataset1.txt", delimiter=' ')
    plot_data_and_gaussians("dataset1.txt", np.zeros((2,)), np.mean(data,0), np.eye(2), np.eye(2))	

 
