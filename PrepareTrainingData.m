function [Training, Group] = PrepareTrainingData(fileList,landmark,param)
%PREPARETRAININGDATA Prepare training data.
% [Training, Group] = PrepareTrainingData(fileList, landmark, param) reads
% images and corresponding landmarks from filelist and landmark files.
% Automatically generate negative samples around positive landmarks.
% Combine positive and negative features into Training and corresponding
% labels into Group, which can be used for further training
%
% Output:
% Training - 5M-by-N-by-K matrix. M corresponds to M positive observations. 
% N is number of dimensions. K is the group number of training data.
%
% Group - 5M-by-1 logical matrix. Each row corresponds to group label
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
% samples arranged in true, false, false, false, false order (1 pos sample + 
% 4 neg sample neighbors)
Group = false(M*5,1);
Group(1:5:end) = true; % mark as true

%% Extract landmarks from image
for idx = 1:M % idxth image
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
    sampleIdx = 1+(idx-1)*5;
    for iMark = 1:K % iMarkth feature
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
        Training(sampleIdx,:,iMark) = featData(:); % pos
        %% Extract negative samples
        % top left 
        topLeft = feat2D - defaultPatchSize;
        bottomRight = feat2D;
        featData = image(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2));
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+1,:,iMark) = featData(:); % top left neg sample
   
        % top right
        topLeft = [feat2D(1),feat2D(2)-defaultPatchSize(2)];
        bottomRight = [feat2D(1)+defaultPatchSize(1),feat2D(2)];
        featData = image(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2));
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+2,:,iMark) = featData(:); % top right neg sample
        
        % bottom left
        topLeft = [feat2D(1)-defaultPatchSize(1),feat2D(2)];
        bottomRight = [feat2D(1),feat2D(2)+defaultPatchSize(2)];
        featData = image(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2));
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+3,:,iMark) = featData(:); % bottom left neg sample
        
        % bottom right
        topLeft = feat2D;
        bottomRight = feat2D+defaultPatchSize;
        featData = image(topLeft(1):bottomRight(1),topLeft(2):bottomRight(2));
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+4,:,iMark) = featData(:); % bottom right neg sample
        
    end % for iMark of landmark
end % for idx of image
