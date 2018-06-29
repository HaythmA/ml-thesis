close all; clear; clc;

% load data (training and test set) from a mat file
load('data.mat');

% Split the original training set into training and validation sets
NTr = length(XTr); p = 0.3;
XCv = XTr(1:p*NTr, :);
YCv = YTr(1:p*NTr);
XTr = XTr(p*NTr+1:NTr, :);
YTr = YTr(p*NTr+1:NTr, :);

% Find HOG features of all the images. Comment this part if feature
% extraction is not required
cellSize = [8 8]; bins = 8; clip = 0;
XTr = hogFeatures(XTr, cellSize, bins, clip);
XCv = hogFeatures(XCv, cellSize, bins, clip);
XTe = hogFeatures(XTe, cellSize, bins, clip);

% Convert the feature matrices to sparse format which is the
% requirement of libsvmtrain
XTr = sparse(XTr);
XCv = sparse(XCv);
XTe = sparse(XTe);

% initiallize some variables for examples C and Gamma as vectors on which
% we will perform cross-validation and select the optimal parameters.
% Train the Gaussian SVM model for all possible values of C and Gamma
% and store the parameters in a cell-matrix
Lc = 6; C = logspace(-10, 0, Lc)';
Lg = 6; Gamma = logspace(-10, 0, Lg);
model = cell(Lc, Lg);
start = tic;
disp(['started at ', num2str(toc(start)/60), ' minutes']);
for lc = 1:Lc;
    disp(lc);
    for lg = 1:Lg;
        options = ['-q -c ', num2str(C(lc)), ' -g ', num2str(Gamma(lg))];
        model{lc, lg} = libsvmtrain(YTr, XTr, options);
        disp(['   ', num2str(lg), ' finished at ', ...
            num2str(toc(start)/60), ' minutes.']);
    end
end

% Find and Store the training and validation errors in the corresponding
% matrices
JTr = zeros(Lc, Lg, 3);
JCv = zeros(Lc, Lg, 3);
for lc = 1:Lc;
    disp(lc);
    for lg = 1:Lg;
        [~, JTr(lc, lg, :), ~] = libsvmpredict(YTr, XTr, ...
            model{lc, lg}, '-q');
        [~, JCv(lc, lg, :), ~] = libsvmpredict(YCv, XCv, ...
            model{lc, lg}, '-q');
        disp(['   ', num2str(lg), ' finished at ', ...
            num2str(toc(start)/60), ' minutes.']);
    end
end

% Choose the parameters C and Gamma with smallest validation error
% Save the results for the report
trainAccuracy = JTr(:, :, 1);
validationAccuracy = JCv(:, :, 1);
[i, j] = find(validationAccuracy == max(validationAccuracy(:)));
modelGaussianSVM1 = model{i, j};
[~, JTe, ~] = libsvmpredict(YTe, XTe, modelGaussianSVM1, '-q');
testAccuracy = JTe(1);
save('GaussianSVM1.mat', 'modelGaussianSVM1', 'trainAccuracy', ...
    'validationAccuracy', 'testAccuracy', 'C', 'Gamma');

