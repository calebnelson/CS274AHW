% Code to read in and display the clown image in MATLAB

z = imread('clown.jpg');  % read in the clown image
imagesc(z);  % display the image

% convert to a 3d matrix of RGB values per pixel...
[d1 d2 d3] = size(z);
r = reshape(z(:,:,1),d1*d2,1);
g = reshape(z(:,:,2),d1*d2,1);
b = reshape(z(:,:,3),d1*d2,1);
rgbdata = [r g b];

% show scatter plots of combinations of r, g, and b channels
plot(r,g,'.')
plot(r,b,'.')
plot(g,b,'.')