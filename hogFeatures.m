function Hogs = hogFeatures(X, cellSize, bins, clip)

if (nargin<4)
    clip = 0;
end

N = size(X, 1);
imgs = reshape(X', 32, 32, N);
imgs = imgs(1+clip:end-clip, 1+clip:end-clip, :);

hogs = extractHOGFeatures(imgs(:, :, 1), ...
        'CellSize', cellSize, 'NumBins', bins);
Hogs = zeros(N, length(hogs));
Hogs(1, :) = hogs;
for k = 2:N;
    Hogs(k, :) = extractHOGFeatures(imgs(:, :, k), ...
        'CellSize', cellSize, 'NumBins', bins);
end


end

