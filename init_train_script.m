% script file to combine whole process 
% written by (C) Shuang Feng, in July 1st, 2013
% load data
if exist('filename.mat','file')
    load filename.mat;
else
    % load from .txt file
    filename = importdata('filelist.txt');
    save('filename.mat',filename);
end
if exist('landmark88.mat','file')
    load landmark88.mat;
else
    % load from .txt file
    landmark88 = importdata('landmark88.txt','file');
    save('landmark88.mat',landmark88);
end

% invoke algorithm function
param = struct('DefaultFaceSize',[128 128],...
                'StdFaceSize',[40 42],...  % derived by the program
                'StdPatchSize',[11 11],... % also
                 'FeatureType', 'intensity'); % avail choices: intensity, LBP, gradient
[w, b, theta, beta] = LandmarkDetectorTrain(filename,landmark88,param);

% save result
[N, K] = size(w); % N - feature dimension. K - number of classifiers

% y(x) = 1./(1+exp(beta*f(x) + theta))
% f(x) = w*x + b
fid_w = fopen('detector.txt','w');
fid_b = fopen('intersect.txt','w');
fid_beta_theta = fopen('ab.txt','w');
for iMark = 1:K
    % w
    fprintf(fid_w, '%g ', w(:,iMark));
    fprintf(fid_w, '\n');
    % beta,thea
    fprintf(fid_beta_theta,'%g %g\n',beta(iMark),theta(iMark));
    % b
    fprintf(fid_b,'%g\n',b(iMark));
end % for each landmark detector
fclose(fid_w);
fclose(fid_b);
fclose(fid_beta_theta);
% save a .mat copy in current workspace
save('train_result.mat',w, b, theta, beta);