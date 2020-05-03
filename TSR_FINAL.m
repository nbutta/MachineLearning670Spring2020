%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% Use this script as a template for implementing individual classifiers.

%% Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% Sensor/ Signal Capture 
% Setup paths to datasets (replace as necessary)

%basePath = 'C:/Users/j39950/Documents/JHU/Courses/2020/670 Machine Learning/Proj/Traffic Sign Recognition/Kaggle/';
basePath = fullfile(fileparts(fullfile(mfilename('fullpath'))),'..','gtsrb-german-traffic-sign/');
trainCsv = 'Train.csv';
testCsv = 'Test.csv';
metaCsv = 'Meta.csv';

%% Data Conditioning

% [OPTIONAL] Eliminate low quality test and training images
% Set resTrim and contrastTrim to 1 if you want to use entire dataset.
[trainPaths, trainWidths, trainHeights, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, trainClasses]  = reduce_dataset(basePath, trainCsv, .75, .75);
[testPaths, testWidths, testHeights, testRoiX1, testRoiY1, testRoiX2, testRoiY2, testClasses]  = reduce_dataset(basePath, testCsv, .75, .75);

% Get the resized images in matrix form. Each row is a grey intensity image
% This captures the ROI before resizing.
% This essentially captures creates our modeling dataset where the pixel intensities
% are the features we are using for each sign.
trainImages = get_images(basePath, trainPaths, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, 50, 50, 'roi');
testImages = get_images(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2, 50, 50, 'roi');

% [OPTIONAL] Do further histogram equalization to improve contrast

%figure;
%imshow(reshape(trainImages(5,:), 50, 50)./255);

trainImagesBoosted = boost_gray_contrast(trainImages);
testImagesBoosted = boost_gray_contrast(testImages);

%figure;
%imshow(reshape(trainImagesBoosted(5,:), 50, 50)./255);

% Information for GTSRB Analysis Tool
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

% [OPTIONAL] Generate new data to reduce class skew

%% Feature Extraction

% Extract Hue Histogram features
train_hue_features = get_hue_histograms(basePath, trainPaths, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, 100);
test_hue_features = get_hue_histograms(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2, 100);

% [NOT WORKING WELL] Extract a "roundness" feature for each image
% roundness_features = detect_roundness(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2);

% Extract PCA features
numBasis = 40;
[eigsigns, eigvals] = pca_basis(trainImagesBoosted, numBasis);

train_pca_features = trainImagesBoosted*eigsigns;
test_pca_features = testImagesBoosted*eigsigns;

[test_known_classes_names, test_known_classes_categ] = classid_to_name(testClasses);

%% Modeling/Prediction KNN
% PCA based prediction
k_neighbors = 5;
%knn_pred_classes = knn_nick(1, train_pca_features, trainClasses, test_pca_features);
knn_pred_classes = knn_predict(k_neighbors,train_pca_features,trainClasses,test_pca_features);

[knn_pred_classes_names, knn_pred_classes_categ] = classid_to_name(knn_pred_classes);

figure;
%[C, order] = confusionmat(testClasses, knn_predicted_classes_pca);
cm = confusionchart(test_known_classes_names, knn_pred_classes_names, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['KNN - PCA Basis: ',num2str(numBasis), ' k-neighbors: ',num2str(k_neighbors)]);

figure;
%[C, order] = confusionmat(testClasses, knn_predicted_classes_pca);
cm = confusionchart(test_known_classes_categ, knn_pred_classes_categ, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['KNN Categories - PCA Basis: ',num2str(numBasis), ' k-neighbors: ',num2str(k_neighbors)]);

% check the performance of the model
cp = classperf(test_known_classes_categ,knn_pred_classes_categ);
fprintf('KNN Categories - PCA Basis: %d k-neighbors: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    k_neighbors,...
    cp.CorrectRate,cp.ErrorRate);
cp = classperf(test_known_classes_names,knn_pred_classes_names);
fprintf('KNN - PCA Basis: %d k-neighbors: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    k_neighbors,...
    cp.CorrectRate,cp.ErrorRate);

% write output for GTSRB Analysis Tool
generate_tsrb_results('KNN_Results.csv',signstrain,signstest,knn_pred_classes);

%% Modeling/Prediction SVM
% Train the model
mdl = svm_train(train_pca_features, trainClasses);

% Test the model
svm_pred_labels = svm_predict(mdl,test_pca_features);

[svm_pred_classes_names, svm_pred_classes_categ] = classid_to_name(svm_pred_labels);


figure;
cm = confusionchart(test_known_classes_names, svm_pred_classes_names, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['SVM - PCA Basis: ',num2str(numBasis)]);

figure;
cm = confusionchart(test_known_classes_categ, svm_pred_classes_categ, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';
title(['SVM Categories - PCA Basis: ',num2str(numBasis)]);

% check the performance of the model
cp = classperf(test_known_classes_categ,svm_pred_classes_categ);
fprintf('SVM Categories - PCA Basis: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    cp.CorrectRate,cp.ErrorRate);
cp = classperf(test_known_classes_names,svm_pred_classes_names);
fprintf('SVM - PCA Basis: %d CorrectRate: %f ErrorRate: %f \n',...
    numBasis,...
    cp.CorrectRate,cp.ErrorRate);

% write output for GTSRB Analysis Tool
generate_tsrb_results('SVM_Results.csv',signstrain,signstest,svm_pred_labels);

% Example of KNN classification using hue features
knn_predicted_classes_hue = knn_nick(1, train_hue_features, trainClasses, test_hue_features);
%sum(knn_predicted_classes_hue~=testClasses)/length(testClasses)

[knn_pred_classes_names, knn_pred_classes_categ] = classid_to_name(knn_predicted_classes_hue);
%[test_known_classes_names, test_known_classes_categ] = classid_to_name(testClasses);

fig = figure;
%[C, order] = confusionmat(testClasses, knn_predicted_classes_pca);
cm = confusionchart(test_known_classes_names, knn_pred_classes_names, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';

fig = figure;
%[C, order] = confusionmat(testClasses, knn_predicted_classes_pca);
cm = confusionchart(test_known_classes_categ, knn_pred_classes_categ, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';

%% Modeling/Prediction CNN

%% Modeling/Prediction Naive Bayes

%% Ignore this for now.. Multi tiered classification  

% trans = [0, 0, 0, ...
%         0, 0, 0, ...
%         1, 0, 0, ...
%         0, 0, 3, ...
%         5, 3, 4, ...
%         3, 0, 4, ...
%         3, 3, 3, ...
%         3, 5, 3, ...
%         3, 5, 3, ...
%         3, 3, 3, ...
%         3, 3, 1, ...
%         6, 6, 6, ...
%         6, 6, 6, ...
%         6, 6, 1, ...
%         1];
% 
%  known = trans(testClasses+1);
%  predicted = trans(closest_sign+1);
%  
%  sum(known~=predicted)/num_test_images
%  o = [];
%  
%  for i = 1:43
%     r = testClasses==i-1;
%     o = [o sum(known(r)~=predicted(r))/num_test_images];
%  end
%  
%  bar(o)
%  
% fig = figure;
% [C, order] = confusionmat(known, predicted);
% cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');
% 
% % This will sort based on the true positive rate
% cm.Normalization = 'row-normalized'; 
% sortClasses(cm,'descending-diagonal')
% cm.Normalization = 'absolute';