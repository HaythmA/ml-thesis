function Hogs = hogFeatures(X, cellSize, bins, clip)

% Inputs;
% X - Matrix of Images of order N by D where N is number of images and D is
% number of pixels
% cellSize = a 2-D vector which contains the cellSize. a parameter needed
% for extractHOGFeatures method
% bins - number of histogram bins. a parameter needed for
% extractHOGFeatures method
% clip - optional parameter, the method clips the image from all four
% boundaries by number of pixels given be clip
% 
% Output;
% Hogs - Matrix of HOG Features of Order N by Dh where N is number of
% images and Dh is number of Hog features for each image. Dh
% depends on D, cellSize and bins.

if (nargin<4)
    clip = 0;
end

% Reshape the original Matrix to a 3D tensor of order N by d by d. where d
% is the number of rows and number of columns of each image.
[N, D] = size(X);
d = sqrt(D);
imgs = reshape(X', d, d, N);
imgs = imgs(1+clip:end-clip, 1+clip:end-clip, :);

% Extract HOG features
hogs = extractHOGFeatures(imgs(:, :, 1), ...
        'CellSize', cellSize, 'NumBins', bins);
Hogs = zeros(N, length(hogs));
Hogs(1, :) = hogs;
for k = 2:N;
    Hogs(k, :) = extractHOGFeatures(imgs(:, :, k), ...
        'CellSize', cellSize, 'NumBins', bins);
end


end

