function  plot_gauss_parameters(mu, covar,xaxis,yaxis,colorstr)
% PLOT_GAUSS:  plot_gauss_parameters(mu, covar,xaxis,yaxis,colorstr)
%
%  MATLAB function to plot the covariance of a 2-dimensional Gaussian
%  model as a "3-sigma" covariance ellipse  
%                                       CS 274 Demo Function, Winter 2019 
%
%  INPUTS: 
%   mu: the d-dimensional mean vector of a Gaussian model
%   covar: d x d matrix: the d x d covariance matrix of a Gaussian model
%   xaxis: an integer between 1 and d indicating which of the features is 
%         to be used as the x axis
%   yaxis: another integer between 1 and d for the y axis
%   colorstr: string defining the color of the ellipse plotted (e.g., 'r')


% Extract the relevant dimensions from the ith component matrix
covar2d = [covar(xaxis,xaxis) covar(xaxis,yaxis); covar(yaxis,xaxis) covar(yaxis,yaxis)];


% Use some results from standard geometry to figure out the ellipse
% equations from the covariance matrix. 
 icov = inv(covar2d);
 a = icov(1,1);
 c = icov(2,2); 
 b = icov(1,2)*2 + 0.0000001; 
 theta = 0.5*acot( (a-c)/b);

 sc = sin(theta)*cos(theta);
 c2 = cos(theta)*cos(theta);
 s2 = sin(theta)*sin(theta);

 a1 = a*c2 + b*sc + c*s2;
 c1 = a*s2 - b*sc + c*c2;

 th= 0:2*pi/100:2*pi;

 
% Calculate contours for the 2d normals at 3-sigma away from the mean mu
% (equivalently, Mahalanobis dist = constant (corresponding to 3 sigma))
mhdist = 3;
 x1 = sqrt(mhdist/a1)*cos(th);
 y1 = sqrt(mhdist/c1)*sin(th);
 
 x = x1*cos(theta) - y1*sin(theta) + mu(xaxis);
 y = x1*sin(theta) + y1*cos(theta) + mu(yaxis);
% plot the ellipse 
 plot(x,y,colorstr,'LineWidth',2);


	

 
