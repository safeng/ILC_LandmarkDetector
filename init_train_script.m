% script file to combine whole process 
% written by (C) Shuang Feng, in July 1st, 2013
% load data. Suppose we haved loaded from text file
load filename.mat;
load landmark88.mat;

% invoke algorithm function
param = struct('DefaultFaceSize',[256 256],...
                'StdFaceSize',[40 42],... % set by the program
                'StdPatchSize',[11 11],...
                 'FeatureType', 'intensity');
[w, b, theta, beta] = LandmarkDetectorTrain(filename,landmark88);

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
