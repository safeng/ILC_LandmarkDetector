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
% See also:
% lbp.m

%% Check field validity
TF = isfield(param,{'DefaultFaceSize','StdFaceSize','StdPatchSize','FeatureType'});
if ~all(TF)
    error('Missing fields');
end
defaultSize = param.DefaultFaceSize;
stdFaceSize = param.StdFaceSize;
stdPatchSize = param.StdPatchSize;
scaleFactor = defaultSize./stdFaceSize;
%defaultPatchSize = round(stdPatchSize.*scaleFactor); 
defaultPatchSize = stdPatchSize; 
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

%% Extract landmarks from image
M = 100;
for idx = 1:M % idxth image
    image = imread(fileList{idx});
    % color image
    if ndims(image)==3
       image = rgb2hsv(image);
       image = image(:,:,3); % extract intensity channel
    end
    image = double(image);
    %% Generate feature image from original gray-scale image
    sp = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1]; % neighbor loc relative to central pix
    hx = [1;0;-1];hy = [1 0 -1]; % sobel edge filter
    if strcmp(param.FeatureType,'intensity')
        % do nothing
    elseif strcmp(param.FeatureType,'LBP')
        % add padding border around image
        image = padarray(image,[1 1]);
        % generate LBP image with the following pattern
        %   1   2   3
        %   4   o   5
        %   6   7   8
        % 1 the lowest digit. 8 is the highest digit
        image = lbp(image,sp,0,'i'); % lbp image
        
    elseif strcmp(param.FeatureType,'gradient')
        imgx = imfilter(image,hx);
        imgy = imfilter(image,hy);
        image = imgx.^2 + imgy.^2; % use engergy terms as gradient image
       
    else
        error('Unknown feature types');
    end
    %% Extract features indexed by feature locations
    featLoc = landmark(idx,:); % corresponding K feature locations
    featLoc = reshape(featLoc,2,K); % get K-by-2 matrix
    featLoc = featLoc';
    sampleIdx = 1+(idx-1)*5; % index of current positive sample
    img2show = ones(size(image))*255; % img to show
    
    for iMark = 1:K % iMarkth feature
        % extract image patch centered at each landmark
        feat2D = featLoc(iMark,:);
        patchHalfSize = round(defaultPatchSize/2);
        %% Extract positive samples
        topLeft = feat2D - patchHalfSize;
        bottomRight = feat2D + patchHalfSize;
        featData = image(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)); % feature matrix
        %img2show(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)) = featData;
        % resize to standard size
        featData = imresize(featData, stdPatchSize);
        % store positive sample in training data matrix
        Training(sampleIdx,:,iMark) = featData(:); % pos
        Group(sampleIdx) = true;
        
        %% Extract 4 negative samples around positive sample
        % top left 
        topLeft = round(feat2D - defaultPatchSize - patchHalfSize);
        bottomRight = round(feat2D - patchHalfSize);
        featData = image(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1));
        %img2show(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)) = featData;
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+1,:,iMark) = rand(size(featData(:))); % top left neg sample
        
        % top right
        topLeft = round([feat2D(1)+patchHalfSize(1),feat2D(2)-defaultPatchSize(2)*1.5]);
        bottomRight = round([feat2D(1)+defaultPatchSize(1)*1.5,feat2D(2)+patchHalfSize(2)]);
        featData = image(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1));
        %img2show(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)) = featData;
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+2,:,iMark) = rand(size(featData(:))); % top right neg sample
        
        % bottom left
        topLeft = round([feat2D(1)-defaultPatchSize(1)*1.5,feat2D(2)+patchHalfSize(2)]);
        bottomRight = round([feat2D(1)-patchHalfSize(1),feat2D(2)+defaultPatchSize(2)*1.5]);
        featData = image(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1));
        %img2show(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)) = featData;
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+3,:,iMark) = rand(size(featData(:))); % bottom left neg sample
        
        % bottom right
        topLeft = round(feat2D + patchHalfSize);
        bottomRight = round(feat2D + defaultPatchSize + patchHalfSize);
        featData = image(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1));
        %img2show(topLeft(2):bottomRight(2),topLeft(1):bottomRight(1)) = featData;
        featData = imresize(featData, stdPatchSize);
        Training(sampleIdx+4,:,iMark) = rand(size(featData(:))); % bottom right neg sample
    end % for iMark of landmark
    % show image
%     subplot(1,2,1);
%     imshow(img2show,[]);
%     subplot(1,2,2);
%     imshow(image,[]); % original image
%     hold on;
%     % draw landmarks
%     plot(featLoc(:,1),featLoc(:,2),'r.');
%     hold off;
    if idx>1
        fprintf(repmat('\b',1,5+1+length('complete')));
    end
    fprintf('%5.2f%%complete',idx/M*100);
end % for idx of image
fprintf('\n');