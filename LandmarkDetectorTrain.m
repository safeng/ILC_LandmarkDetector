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
RealOutput = zeros(M,K); % M samples, K classifiers
for iMark = 1:K
    linearSVM = SVMStruct{iMark};
    SupportVectors = linearSVM.SupportVectors; % suport vectors
    Alpha = linearSVM.Alpha; % weight of support vectors
    ScaleData = linearSVM.ScaleData;
    
    % test correctness of training
    labels = svmclassify(linearSVM,Training(:,:,iMark));
    missClass = sum(labels~=Group(:,iMark)); % number of misclassified samples
    fprintf(1, 'Error rate %g of %d svm\n', missClass/M, iMark);
end
%% Logistic regression
