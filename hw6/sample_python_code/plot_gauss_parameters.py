import numpy as np
import pylab as P
import matplotlib.mlab as mlab



def  plot_gauss_parameters(mu, covar, colorstr, delta=.1):

    '''
    %PLOT_GAUSS:  plot_gauss_parameters(mu, covar,xaxis,yaxis,colorstr)
    %
    %  Python function to plot the covariance of a 2-dimensional Gaussian
    %  model as a "3-sigma" covariance ellipse  
    %
    %  INPUTS: 
    %   mu: the d-dimensional mean vector of a Gaussian model
    %   covar: d x d matrix: the d x d covariance matrix of a Gaussian model
    %   colorstr: string defining the color of the ellipse plotted (e.g., 'r')
    '''
    
    # make grid
    x = np.arange(mu[0]-3.*np.sqrt(covar[0,0]), mu[0]+3.*np.sqrt(covar[0,0]), delta)
    y = np.arange(mu[1]-3.*np.sqrt(covar[1,1]), mu[1]+3.*np.sqrt(covar[1,1]), delta)
    X, Y = np.meshgrid(x, y)

    # get pdf values
    Z = mlab.bivariate_normal(X, Y, np.sqrt(covar[0,0]),  np.sqrt(covar[1,1]), mu[0], mu[1], sigmaxy=covar[0,1])

    P.contour(X, Y, Z, colors=colorstr, linewidths=4)

#plot_gauss_parameters(3+np.zeros((2,)), .5*np.eye(2), 'r')
#P.show()

	

 
