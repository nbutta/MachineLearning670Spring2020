%%
% File: Traffic_Sign_KNN.m
%   Load the tranining and test data sets. Use the k-nearest neighbor
%   classifier to test the model. 
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
trainCsv = 'Train.csv';
testCsv = 'Test.csv';
metaCsv = 'Meta.csv';

%% 2. Data Conditioning

% [OPTIONAL] Eliminate low quality test and training images
% Set resTrim and contrastTrim to 1 if you want to use entire dataset.
[trainPaths, trainWidths, trainHeights, ...
    trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2,...
        trainClasses]  = reduce_dataset(sBasePath, trainCsv, .75, .75);
    
[testPaths, testWidths, testHeights, ...
    testRoiX1, testRoiY1, testRoiX2, testRoiY2,...
        testClasses]  = reduce_dataset(sBasePath, testCsv, .75, .75);

% Get the resized images in matrix form. Each row is a grey intensity image
% This captures the ROI before resizing.
% This essentially captures creates our modeling dataset where the pixel intensities
% are the features we are using for each sign.
trainImages = get_images(sBasePath, trainPaths, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, 50, 50, 'roi');
testImages = get_images(sBasePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2, 50, 50, 'roi');

% [OPTIONAL] Do further histogram equalization to improve contrast
trainImagesBoosted = boost_gray_contrast(trainImages);
testImagesBoosted = boost_gray_contrast(testImages);


signstrain.classes = trainClasses;
signstrain.paths = trainPaths;
signstrain.widths = trainWidths;
signstrain.heights = trainHeights;
signstrain.roiX1 = trainRoiX1;
signstrain.roiY1 = trainRoiY1;
signstrain.roiX2 = trainRoiX2;
signstrain.roiY2 = trainRoiY2;

signstest.classes = testClasses;
signstest.paths = testPaths;
signstest.widths = testWidths;
signstest.heights = testHeights;
signstest.roiX1 = testRoiX1;
signstest.roiY1 = testRoiY1;
signstest.roiX2 = testRoiX2;
signstest.roiY2 = testRoiY2;


%% 3. Feature Extraction

% Extract PCA features
numBasis = 40;
[eigsigns, eigvals] = pca_basis(trainImagesBoosted, numBasis);

train_pca_features = trainImagesBoosted*eigsigns;
test_pca_features = testImagesBoosted*eigsigns;

%% 4. Test the model
k_neighbors = 5;
% Predict labels
knn_pred_classes = knn_predict(k_neighbors,...
    train_pca_features, trainClasses, test_pca_features);

[knn_pred_classes_names, knn_pred_classes_categ] = classid_to_name(knn_pred_classes);
[test_known_classes_names, test_known_classes_categ] = classid_to_name(testClasses);

% check the performance of the model
cp = classperf(testClasses,knn_pred_classes);

fprintf('KNN - PCA Basis: %d k-neighbors: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    k_neighbors,...
    cp.CorrectRate,cp.ErrorRate);

cp = classperf(test_known_classes_categ,knn_pred_classes_categ);

fprintf('KNN Categories - PCA Basis: %d k-neighbors: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    k_neighbors,...
    cp.CorrectRate,cp.ErrorRate);

figure;
cm = confusionchart(test_known_classes_names, knn_pred_classes_names, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['KNN - PCA Basis: ',num2str(numBasis), ' k-neighbors: ',num2str(k_neighbors)]);

figure;
cm = confusionchart(test_known_classes_categ, knn_pred_classes_categ, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['KNN Categories - PCA Basis: ',num2str(numBasis), ' k-neighbors: ',num2str(k_neighbors)]);

% write output for GTSRB Analysis Tool
generate_tsrb_results('KNN_Results.csv',signstrain,signstest,knn_pred_classes);

