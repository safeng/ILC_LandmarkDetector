function SVMStruct = SVM_Train(Training,Group)
% SVM_TRAIN train several linear SVMs based on the third dimension of
% training data.
% SVMStruct = SVM_Train(Training,Group) receives training data and
% corresponding labels. Returns SVMStruct struct vectors for each landmark
%
% Output:
% SVMStruct - cell array of svm structures
%
% Input:
% Training - training data 
% Group - labeling data
% 
% See also:
% PrepareTrainingData

%% Check dimensionality of training data
if ndims(Training)~=3
    error('Internal error');
end
[M,~,K] = size(Training); % M/5 images, N is feature dimension, K is the number of svms to be trained
if mod(M,5)
    error('Incorrect number of positive & negative samples');
end
SVMStruct = cell(1,K);
%% Train linear svms for each landmark
for iMark = 1:K
    SVMStruct{iMark} = svmtrain(Training(:,:,iMark),Group);
    if iMark >1
        fprintf(repmat('\b',1,4+1+length(complete)));
    end
    fprintf('%4.1f%%complete.',iMark/K*100);
end
fprintf('\n');