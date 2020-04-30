%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

%% Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% Setup

basePath = 'C:/Users/j39950/Documents/JHU/Courses/2020/670 Machine Learning/Proj/Traffic Sign Recognition/Kaggle/'; 
trainCsv = 'Train.csv';
testCsv = 'Test.csv';
metaCsv = 'Meta.csv';

%% Data Conditioning

% [OPTIONAL]Only use the highest quality training and test images
[trainPaths, trainWidths, trainHeights, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, trainClasses]  = reduce_dataset(basePath, trainCsv, .75, .75);
[testPaths, testWidths, testHeights, testRoiX1, testRoiY1, testRoiX2, testRoiY2, testClasses]  = reduce_dataset(basePath, testCsv, .75, .75);

% [OPTIONAL] Do further histogram equalization?

% [OPTIONAL] Generate new data to reduce class skew

%% Feature Extraction

% Extract Hue Histogram features
train_hue_features = get_hue_histograms(basePath, trainPaths, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, 100);
test_hue_features = get_hue_histograms(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2, 100);

% [NOT WORKING WELL] Extract a "roundness" feature for each image
% roundness_features = detect_roundness(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2);

% Extract PCA features

trainImages = get_images(basePath, trainPaths, trainRoiX1, trainRoiY1, trainRoiX2, trainRoiY2, 50, 50);
testImages = get_images(basePath, testPaths, testRoiX1, testRoiY1, testRoiX2, testRoiY2, 50, 50);

[eigsigns, eigvals] = pca_basis(trainImages, 40);

train_pca_features = trainImages*eigsigns;
test_pca_features = testImages*eigsigns;

%% Modeling/Prediction

% PCA based prediction
knn_predicted_classes_pca = knn_nick(1, train_pca_features, trainClasses, test_pca_features);

fig = figure;
[C, order] = confusionmat(testClasses, knn_predicted_classes_pca);
cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';

% Example of KNN classification using hue features
knn_predicted_classes_hue = knn_nick(3, train_hue_features, trainClasses, test_hue_features);
sum(knn_predicted_classes_hue~=testClasses)/length(testClasses)

%% Ignore this for now.. Multi tiered classification  

trans = [0, 0, 0, ...
        0, 0, 0, ...
        1, 0, 0, ...
        0, 0, 3, ...
        5, 3, 4, ...
        3, 0, 4, ...
        3, 3, 3, ...
        3, 5, 3, ...
        3, 5, 3, ...
        3, 3, 3, ...
        3, 3, 1, ...
        6, 6, 6, ...
        6, 6, 6, ...
        6, 6, 1, ...
        1];

 known = trans(testClasses+1);
 predicted = trans(closest_sign+1);
 
 sum(known~=predicted)/num_test_images
 o = [];
 
 for i = 1:43
    r = testClasses==i-1;
    o = [o sum(known(r)~=predicted(r))/num_test_images];
 end
 
 bar(o)
 
fig = figure;
[C, order] = confusionmat(known, predicted);
cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';