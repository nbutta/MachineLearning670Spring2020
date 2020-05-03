%% Cassie Xia
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

%% Naive Bayes - Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% 1. Load the training data.
sBasePath = fullfile(fileparts(fullfile(mfilename('fullpath'))),'..','gtsrb-german-traffic-sign/');

sTrainingPath = [sBasePath, 'Train.csv'];

% Find signstrain.mat and signstest.mat
% If not found generate them
curDir = pwd;
filename = [curDir,'/','signstrain.mat'];

if isfile(filename)
    signstrain = load(filename);
else
    % generate matfile
    signstrain = generate_csv2mat(sTrainingPath,filename);
end

% Load the test data

sTestPath = [sBasePath, 'Test.csv'];

filename = [curDir,'/','signstest.mat'];

if isfile(filename)
    signstest = load(filename);
else
     % generate matfile
     signstest = generate_csv2mat(sTestPath,filename);
end

% Training data
tr_images = signstrain.A;
tr_labels = signstrain.classes;

% Test data
test_images = signstest.A;
test_labels = signstest.classes;

% Image dimensions
M = 50;
N = 50;
num_pixels = M*N;

num_signs = 43;

%% Use PCA
% Perform dimensionality reduction
numBasis = 40;
[V,D] = pca_basis(tr_images,numBasis);

% Produce a column vector of eigenvalues
D = diag(D);

% Projections
train_projection = tr_images*V;
test_projection = test_images*V;

%% 2. Naive Bayes on first 100 eigenvalues of PCA

% m - mean: 1xc matrix, jth column is mean of the jth class
% S - 1x1xc matrix, covariance matrix of the normal distribution of the jth
% class
% P - c-dimensional vector, a priori probability of jth class
% X - 1xN matrix, columns are the data vectors to be classified

num_training_data = size(tr_labels,1);
num_features = 120;  % switch around

P = zeros(1,num_signs);
S = zeros(num_features,num_features,num_signs);
m = zeros(num_features,num_signs);
for i = 0:(num_signs-1)
    % P: c-dimensional vector, whose j-th component is the a priori
    % probability of the j-th class.
    P(i+1) = sum(tr_labels == i)/num_training_data; 
    % TODO: double check on dimensions for mean and covariance
    % m - mean: 1xc matrix, jth column is mean of the jth class
    m(:,i+1) = mean(tr_images(tr_labels'==i,1:num_features))';
    S(:,:,i+1) = cov(tr_images(tr_labels'==i,1:num_features))';
end

train_label_zb=bayes_classifier(m,S,P,train_projection(:,1:num_features)');
test_label_zb=bayes_classifier(m,S,P,test_projection(:,1:num_features)');

%% Compare with testClasses
% Note: Very inaccurate, set num_features and numBasis to 50 to run it
% faster. Also comment out train_label_zb for faster run.

%% uncomment when prediction portion is implemented
% check the performance of the model
cp = classperf(signstest.classes,test_label_zb);

fprintf('Bayes - PCA Basis: %d CorrectRate: %f ErrorRate: %f \n',...
    num_features,...
    cp.CorrectRate,cp.ErrorRate);

% Performance of training data
tcp = classperf(signstrain.classes,train_label_zb);

fprintf('Train Bayes - PCA Basis: %d CorrectRate: %f ErrorRate: %f \n',...
    num_features,...
    tcp.CorrectRate,tcp.ErrorRate);

fig = figure;
[C, order] = confusionmat(signstest.classes, test_label_zb);
cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['Bayes - PCA Basis: ',num2str(40)]);

% write output for GTSRB Analysis Tool
generate_tsrb_results('Bayes_Results.csv',signstrain,signstest,test_label_zb);
