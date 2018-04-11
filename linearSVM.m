close all; clear; clc;

load('data.mat');

NTr = length(XTr); p = 0.3;
XCv = XTr(1:p*NTr, :);
YCv = YTr(1:p*NTr);
XTr = XTr(p*NTr+1:NTr, :);
YTr = YTr(p*NTr+1:NTr, :);

cellSize = [8 8]; bins = 8; clip = 0;
XTr = hogFeatures(XTr, cellSize, bins, clip);
XCv = hogFeatures(XCv, cellSize, bins, clip);
XTe = hogFeatures(XTe, cellSize, bins, clip);

XTr = sparse(XTr);
XCv = sparse(XCv);
XTe = sparse(XTe);

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

trainAccuracy = JTr(:, 1);
validationAccuracy = JCv(:, 1);
i = find(validationAccuracy == max(validationAccuracy));
modelHogLinearSVM2 = model{i};
[~, JTe, ~] = libsvmpredict(YTe, XTe, modelHogLinearSVM2, '-q');
testAccuracy = JTe(1);
% save('HogLinearSVM2.mat', 'modelHogLinearSVM2', 'trainAccuracy', ...
%     'validationAccuracy', 'testAccuracy', 'C');

