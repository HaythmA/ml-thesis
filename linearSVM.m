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

% initiallize some variables for examples C as a vector on which
% we will perform cross-validation and select the optimal parameterss.
% Train the Linear SVM model for all possible values of C and store the
% parameters in a cell-vector.
Lc = 6; C = logspace(-10, 0, Lc)';
model = cell(Lc);
start = tic;
disp(['started at ', num2str(toc(start)/60), ' minutes']);
for lc = 1:Lc;
    options = ['-q -c ', num2str(C(lc)), ' -t 0'];
    model{lc} = libsvmtrain(YTr, XTr, options);
    disp([num2str(lc), ' finished at ', ...
        num2str(toc(start)/60), ' minutes.']);
end

% Find and Store the training and validation errors in the corresponding
% vectors
JTr = zeros(Lc, 3);
JCv = zeros(Lc, 3);
for lc = 1:Lc;
    [~, JTr(lc, :), ~] = libsvmpredict(YTr, XTr, ...
        model{lc}, '-q');
    [~, JCv(lc, :), ~] = libsvmpredict(YCv, XCv, ...
        model{lc}, '-q');
    disp([num2str(lc), ' finished at ', ...
        num2str(toc(start)/60), ' minutes.']);
end

% Choose the parameter C with smallest validation error
% Save the results for the report
trainAccuracy = JTr(:, 1);
validationAccuracy = JCv(:, 1);
i = find(validationAccuracy == max(validationAccuracy));
modelHogLinearSVM2 = model{i};
[~, JTe, ~] = libsvmpredict(YTe, XTe, modelHogLinearSVM2, '-q');
testAccuracy = JTe(1);
save('HogLinearSVM2.mat', 'modelHogLinearSVM2', 'trainAccuracy', ...
    'validationAccuracy', 'testAccuracy', 'C');

