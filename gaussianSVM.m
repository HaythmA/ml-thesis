close all; clear; clc;

load('data.mat');

NTr = length(XTr); p = 0.3;
XCv = XTr(1:p*NTr, :);
YCv = YTr(1:p*NTr);
XTr = XTr(p*NTr+1:NTr, :);
YTr = YTr(p*NTr+1:NTr, :);

XTr = sparse(XTr);
XCv = sparse(XCv);
XTe = sparse(XTe);

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

trainAccuracy = JTr(:, :, 1);
validationAccuracy = JCv(:, :, 1);
[i, j] = find(validationAccuracy == max(validationAccuracy(:)));
modelGaussianSVM1 = model{i, j};
[~, JTe, ~] = libsvmpredict(YTe, XTe, modelGaussianSVM1, '-q');
testAccuracy = JTe(1);
save('GaussianSVM1.mat', 'modelGaussianSVM1', 'trainAccuracy', ...
    'validationAccuracy', 'testAccuracy', 'C', 'Gamma');

