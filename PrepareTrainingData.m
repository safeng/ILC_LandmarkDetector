function [Training, Group] = PrepareTrainingData(fileList,landmark,param)
%PREPARETRAININGDATA Prepare training data.
% [Training, Group] = PrepareTrainingData(fileList, landmark, param) reads
% images and corresponding landmarks from filelist and landmark files.
% Automatically generate negative samples around positive landmarks.
% Combine positive and negative features into Training and corresponding
% labels into Group, which can be used for further training
%
% Output:
% Training - M-by-N-by-K matrix. M corresponds to M observations. N is
% number of dimensions. K is the group number of training data.
%
% Group - M-by-K logical matrix. Each row corresponds to group label. K is
% the group number of training data
%
% Input:
% fileList - cell array of image file names
%
% landmark - landmarks for each image
%
% param - structure with following fields
%          Field Name                   Value
%      -----------------    -----------------------------
%      'DefaultFaceSize'    Size of faces in training image 
%      'StdFaceSize'        Size of standard face
%      'StdPatchSize'       Size of patches on standard face
%      'FeatureType'        Feature type string, could be 'intensity'
%                           'LBP' or 'gradient'

%% Check field validity
TF = isfield(param,{'DefaultFaceSize','StdFaceSize','StdPatchSize','FeatureType'});
if ~all(TF)
    error('Missing fields');
end
defaultSize = param.DefaultFaceSize;
stdFaceSize = param.StdFaceSize;
stdPatchSize = param.StdPatchSize;
scaleFactor = defaultSize./stdFaceSize;
% size of patches to be extracted from image
defaultPatchSize = stdPatchSize.*scaleFactor; 

%% Get sizes
M = length(fileList); % number of files
[M_File, K2]  = size(landmark);
if M ~= M_File
    error('Number of files disagrees');
end
K = K2/2; % number of landmarks per face (number of classifiers)
N = prod(stdPatchSize); % feature dimension
Training = zeros(M*5, N, K); % 1 positive samples + 4 negative samples
Group = true(M,K);

%% Extract landmarks from image
for idx = 1:M
    image = imread(fileList{idx});
    % color image
    if ndims(image)==3
       image = rgb2hsv(image);
       image = image(:,:,3); % extract intensity
    end
    %% Generate feature image from original gray-scale image
    if strcmp(param.FeatureType,'intensity')
        % do nothing
    elseif strcmp(param.FeatureType,'LBP')
        
    elseif strcmp(param.FeatureType,'gradient')
        
    else
        error('Known feature types');
    end
    %% Extract features indexed by feature locations
    featLoc = landmark(idx,:); % corresponding K feature locations
    featLoc = reshape(featLoc,K,2); % get K-by-2 matrix
    for iMark = 1:K
        % extract image patch centered at each landmark
        feat2D = featLoc(iMark,:);
        patchHalfSize = floor(defaultPatchSize./2);
        %% Extract positive samples
        topLeft = feat2D-patchHalfSize;
        bottomRight = feat2D+patchHalfSize;
        featData = image(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2)); % feature matrix
        % resize to standard size
        featData = imresize(featData, stdPatchSize);
        % store positive sample in training data matrix
        Training(idx,:,iMark) = featData(:);
        %% Extract negative samples
        % top left 
        topLeft = feat2D - defaultPatchSize;
        bottomRight = feat2D;
        % top right
        topLeft = [feat2D(1),feat2D(2)-defaultPatchSize(2)];
        bottomRight = [feat2D(1)+defaultPatchSize(1),feat2D(2)];
        % bottom left
        topLeft = [feat2D(1)-defaultPatchSize(1),feat2D(2)];
        bottomRight = [feat2D(1),feat2D(2)+defaultPatchSize(2)];
        % bottom right
        topLeft = feat2D;
        bottomRight = feat2D+defaultPatchSize;
    end % for iMark of landmark
end % for idx of image

%% 