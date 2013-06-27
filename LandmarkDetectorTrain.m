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
% fileList : cell array of names of images
% landmark : annotions of facial landmarks
% param : See PrepareTrainingData.m
%
% SEE ALSO:
% PrepareTrainingData SVM_Train logistic

%% Prepare training samples

%% Train linear svms

%% Calculate outputs of trained svms

%% Logistic regression
