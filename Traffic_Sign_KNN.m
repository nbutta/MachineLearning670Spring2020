%%
% File: Traffic_Sign_KNN.m
%   Load the tranining and test data sets. Use the k-nearest neighbor
%   classifier to train and test the model. 
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%   Term: Spring 2020
%   Author: Johanna Rivera
%

%%
% clear workspace
clear all; close all; clc;

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

%% 2. Load the test data

sTestPath = [sBasePath, 'Test.csv'];

filename = [curDir,'/','signstest.mat'];

if isfile(filename)
    signstest = load(filename);
else
     % generate matfile
     signstest = generate_csv2mat(sTestPath,filename);
end


%% 3. Classify data
% Set train and test data
tr_images = signstrain.A;
tr_labels = signstrain.classes;

test_images = signstest.A;
test_labels = signstest.classes;

% Perform dimensionality reduction
[V, D] = pca_basis(tr_images);

% Projections
train_projection = tr_images*V;
test_projection = test_images*V;

k_neighbors = 5;
% Predict labels
p_labels = knn_predict(k_neighbors,train_projection, tr_labels, test_projection);

% check the performance of the model
cp = classperf(test_labels,p_labels);

fprintf('KNN - PCA Basis: %d k-neighbors: %d CorrectRate: %f ErrorRate: %f \n',...
    40,...
    k_neighbors,...
    cp.CorrectRate,cp.ErrorRate);

fig = figure;
[C, order] = confusionmat(test_labels, p_labels);
cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['KNN - PCA Basis: ',num2str(40), ' k-neighbors: ',num2str(k_neighbors)]);

% write output for GTSRB Analysis Tool
generate_tsrb_results('KNN_Results.csv',signstrain,signstest,p_labels);

