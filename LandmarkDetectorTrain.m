function [w, b, theta, beta] = LandmarkDetectorTrain(fileList,landmark,param)
%LANDMARKDETECTORTRAIN trains linear svms and mapps their values to [0,1]
%by logistic regression.
% [w, b, thea, beta] =
% LandmarkDetectorTrain(fileList,landmark,param) trains linear classifiers
% with logistic regression.
%
% Output:
% w - weight vector of linear svms (weighted summation of support vectors).
% N-by-K matrix. N is the feature dimension. K is number of landmarks per
% face.
% b - intercept of linear svm. K dimension row vector.
% theta, beta : E(Y) = 1 ./ ( 1 + exp( beta * X + theta ) ). Both are K
% dimension row vector
% 
% Input:
% fileList - cell array of names of images
% landmark - annotions of facial landmarks
% param - see PrepareTrainingData.m
%
% See also:
% PrepareTrainingData.m     SVM_Train.m
% logistic.m

%% check input
narginchk(1,3);
if nargin == 2
    % prepare a default param
    warning('LandmarkDetectorTrain:missingParam',...
    'Missing size info use the default ones\n');
    param.DefaultFaceSize = [128, 128]; % size of face in training images
    param.StdFaceSize = [40, 42]; % size of face mapped (used in detection)
    param.StdPatchSize = [11, 11];
    param.FeatureType = 'intensity';
end
    
%% Prepare training samples
fprintf('Prepare training data...\n');
[Training, Group] = PrepareTrainingData(fileList, landmark, param);
% M - number of samples per training set, N - feature dimension, K - number
% of landmarks
[M,N,K] = size(Training);

%% Train linear svms
disp('Train linear svms...')
SVMStruct = SVM_Train(Training, Group);

%% Evaluate outputs of trained linear svms
disp('Evaluate trained results...')
b = zeros(1,K); % intersect
w = zeros(N,K); % weight
outputReal = zeros(M,K); % floating-point output of linear svm
compLabels = true(M,K);
for iMark = 1:K
    linearSVM = SVMStruct{iMark};
    b(iMark) = linearSVM.Bias;
    SupportVectors = linearSVM.SupportVectors; % suport vectors (matrix)
    Alpha = linearSVM.Alpha; % weight of support vectors
    ScaleData = linearSVM.ScaleData; % struct of row vectors
    shift = repmat(ScaleData.shift,M,1);
    scaleFactor = repmat(ScaleData.scaleFactor,M,1);
    % w is the weighted summation of support vectors
    weight = Alpha*SupportVectors;
    w(:,iMark) = weight.';
    %% get floating-points values of linear svm output
    % normalize data to zero mean and unit variance
    normData = (Training(:,:,iMark)+shift).*scaleFactor; % M-by-N normalized data matrix
    outputReal(:,iMark) = normData*weight + b(iMark); 
    
    %% test correctness of training and computed real values
    labels = svmclassify(linearSVM,Training(:,:,iMark));
    missClass = sum(labels~=Group(:,iMark)); % number of misclassified samples
    fprintf(1, 'Error rate %g of %d svm\n', missClass/M, iMark);
    % test computed real output
    compLabels(:,iMark) = outputReal>0;
    groupIndex = 1:M;
    fprintf(1,'Unmatched samples:\n_______________\n');
    fprintf(1,'%d\n',groupIndex(compLabels(:,iMark)~=labels));
end

%% Logistic regression
% generate weight factor. 4 for each positive sample, 1 for each negative
% sample
disp('Perform logistic regression...');
weight = ones(M,1);
weight(1:5:M) = 4; % 1 positive sample : 4 negative samples
theta = zeros(1,K);
beta = zeros(1,K);
for iMark = 1:K
    [t,be] = logistic(Training(:,:,iMark),compLabels(:,iMark),weight);
    beta(iMark) = - be;
    theta(iMark) = t;
end