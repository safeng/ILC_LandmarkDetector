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
% SEE ALSO:
% PrepareTrainingData SVM_Train logistic

if vargin == 2
    % prepare a default param
    param.DefaultFaceSize;
    param.StdFaceSize;
    param.StdPatchSize = [11, 11];
    param.FeatureType = 'intensity';
end
    
%% Prepare training samples
[Training, Group] = PrepareTrainingData(fileList, landmark, param);
% M - number of samples per training set, N - feature dimension, K - number
% of landmarks
[M,N,K] = size(Training);

%% Train linear svms
SVMStruct = SVM_Train(Training, Group);

%% Evaluate outputs of trained linear svms
b = zeros(1,K); % intersect
w = zeros(N,K); % weight
outputReal = zeros(M,K); % floating-point output of linear svm
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
    compLabels = outputReal>0;    
end

%% Logistic regression
for iMark = 1:K
    
end
   