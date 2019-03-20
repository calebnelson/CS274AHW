import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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

def plot_data_and_gaussians(data, mu1, mu2, covar1, covar2):
    
    # plot data as a scatter plot
    P.scatter(data[:,0], data[:,1], s=20, c='k', marker='x', alpha=.65, linewidths=2)

    # plot gaussian #1
    plot_gauss_parameters(mu1, covar1, 'r')

    # plot gaussian #2
    plot_gauss_parameters(mu2, covar2, 'b')

    P.show()

def plot_data_and_gaussians3(data, mu1, mu2, mu3, covar1, covar2, covar3):
    
    # plot data as a scatter plot
    P.scatter(data[:,0], data[:,1], s=20, c='k', marker='x', alpha=.65, linewidths=2)

    # plot gaussian #1
    plot_gauss_parameters(mu1, covar1, 'r')

    # plot gaussian #2
    plot_gauss_parameters(mu2, covar2, 'b')

    # plot gaussian #3
    plot_gauss_parameters(mu3, covar3, 'g')

    P.show()

def gaussian_mixture(data, K, init_method, epsilon, niterations, plotflag, RSEED=123):
    '''

        % NOTE that this description is from Matlab, data structures might not correspond to Python
        %
    % Template for a function to fit  a Gaussian mixture model via EM
    % using K mixture components. The data are contained in the N x d "data" matrix.
    %
    % INPUTS
    %  data: N x d real-valued data matrix
    %  K: number of clusters (mixture components)
    %  initialization_method: 1 for memberships, 2 for parameters, 3 for kmeans
    %  epsilon: convergence threshold used to detect convergence
    %  niterations (optional): maximum number of iterations to perform (default 500)
    %  plotflag (optional): equals 1 to plot parameters during learning,
    %                       0 for no plotting (default is 0)
    %  RSEED (optional): initial seed value for the random number generator
    %
    %
    % OUTPUTS
    %  gparams: K-dim structure array containing the learned mixture model parameters:
    %           gparams(k).weight = weight of component k
    %           gparams(k).mean = d-dimensional mean vector for kth component
    %           gparams(k).covariance = d x d covariance vector for kth component
    %  memberships: N x K matrix of probability memberships for "data"
    %
    %  Note: Interpretation of gparams and memberships:
    %
    %    - gparams(k).weight is the probability that a randomly selected row
    %         belongs to component (or cluster) i (so it is "cluster size")
    %
    %    - memberships(i,k) = p(cluster k | x) which is the probability
    %         (computed via Bayes rule) that vector x was generated by cluster
    %         k, according to the "generative" probabilistic model.

    '''
    D = data.shape[1]
    if(init_method == 3):
        # initialize the means to random values
        clusters = data[:K, :]
        print("Kmeans")
        iter = 0
        cluster_assignments = [[] for i in range(K)]
        input_classes = [-1 for point in data]
        while(True):
            cluster_assignments = [[] for i in range(K)]
            # assignment step
            iter += 1
            for i in range(len(data)):
                distances = [np.linalg.norm(data[i] - cluster)
                                            for cluster in clusters]
                cluster_assignments[np.argmin(distances)].append(data[i])
                input_classes[i] = np.argmin(distances)
            print(np.square(np.sum(distances)))

            # relocation step
            previous_clusters = np.copy(clusters)
            for i in range(K):
                a = np.array(cluster_assignments[i])
                clusters[i] = np.mean(a, axis=0)

            # if clusters are the same as last time, we've converged
            if(np.array_equal(previous_clusters, clusters)):
                print("Converged after " + str(iter) + " iterations")
                break
            if(iter >= niterations):
                print("Hit max iterations")
                break

            # print "After iter " + str(iter)
        if (plotflag == 1):
            plt.scatter(x=data[:, 0], y=data[:, 1], s=20, c=input_classes)
            plt.scatter(x=clusters[:, 0], y=clusters[:, 1], s=100)
            plt.show()
        return (clusters, cluster_assignments)

    # initialize....
    print("EM")
    N = len(data)  # N is the number of training examples
    means = np.random.randn(K, data.shape[1])
    weights = np.random.randn(N, K)
    covs = np.zeros((K, D, D))
    for ki in range(K):
        covs[ki] = np.identity(D)
    mixings = np.full(K, 1.0/K)
    iter = 0
    logl = 0
    if(plotflag == 1):
        if (K == 3):
            plot_data_and_gaussians3(data, means[0], means[1], means[2], covs[0], covs[1], covs[2])
        else:
            plot_data_and_gaussians(data, means[0], means[1], covs[0], covs[1])
    while(True):
        iter += 1
        # perform E-step...
        for ni in range(N):
            for ki in range(K):
                weights[ni][ki] = mixings[ki] * \
                    multivariate_normal.pdf(
                        data[ni], mean=means[ki], cov=covs[ki])
            weights[ni] /= np.sum(weights[ni])

        Nk = np.zeros(K)
        for ki in range(K):
            Nk[ki] = np.sum(weights.T[ki])

        # perform M-step...
        mixings = np.divide(Nk, N)
        means = np.dot(weights.T, data)
        for ki in range(K):
            means[ki] = np.divide(means[ki], Nk[ki])

        for ki in range(K):
            covs[ki] = np.zeros((D, D))
            for ni in range(N):
                arr = np.array(data[ni] - means[ki]).reshape(D, 1)
                covs[ki] += np.dot(weights[ni][ki], np.dot(arr, arr.T))
            covs[ki] = np.divide(covs[ki], Nk[ki])
            antisingular = np.zeros(covs[ki].shape)
            np.fill_diagonal(antisingular, 1e-10)
            covs[ki] += antisingular

        # compute log-likelihood and print to screen.....
        previous_logl = logl
        logl = 0
        for ni in range(N):
            loglk = 0
            for ki in range(K):
                loglk += mixings[ki] * \
                    multivariate_normal.pdf(
                        data[ni], mean=means[ki], cov=covs[ki])
            logl += np.log(loglk)
        print(logl)
        # print("Diff from previous LogL " + str(previous_logl - logl))

        # check for convergence.....
        if (np.absolute(previous_logl - logl) < epsilon):
            print("stopping criterion met after " +
                    str(iter) + " iterations")
            break
        if (iter > niterations):
            print("max iterations reached")
            break
    if(plotflag == 1):
        if (K == 3):
            plot_data_and_gaussians3(data, means[0], means[1], means[2], covs[0], covs[1], covs[2])
        else:
            plot_data_and_gaussians(data, means[0], means[1], covs[0], covs[1])
    return (means, covs, logl)

def BIC(data, maxK, epsilon, niterations=500, plotflag=0, RSEED=123):
    maxBIC = -1e10
    chosenK = -1
    for k in range(1, maxK+1):
        print(k)
        EM = gaussian_mixture(data, k, 1, epsilon, niterations, plotflag, RSEED)
        print(EM)
        pk = (k - 1) + (2 * k) + (3 * k)
        bic = EM[2] - pk * .5 * np.log(len(data))
        print(bic)
        if (bic > maxBIC):
            maxBIC = bic
            chosenK = k
    return (chosenK, maxBIC)


if __name__ == "__main__":
    for filename in ["dataset1.txt", "dataset2.txt", "dataset3.txt"]:
        data = np.genfromtxt(filename)
        if (filename=="dataset2.txt"):
            clusters = gaussian_mixture(data, 3, 3, 0.1, 500, 1)
            print(clusters[0])
            EM = gaussian_mixture(data, 3, 1, 0.001, 500, 1)
            print(EM)
            print(BIC(data, 5, 0.001))
        else:
            clusters = gaussian_mixture(data, 2, 3, 0.1, 500, 1)
            print(clusters[0])
            EM = gaussian_mixture(data, 2, 1, 0.001, 500, 1)
            print(EM)
            print(BIC(data, 5, 0.001))
