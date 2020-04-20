%% Cassie Xia
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

%% Naive Bayes - Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% 1. Load the training data.

%sBasePath = 'C:/Users/j39950/Documents/JHU/Courses/2020/670 Machine Learning/Proj/Traffic Sign Recognition/Kaggle/'; 
sBasePath = fullfile(fileparts(fullfile(mfilename('fullpath'))),'..','gtsrb-german-traffic-sign/');

sTrainingPath = [sBasePath, 'Train.csv'];
sTestPath = [sBasePath, 'Test.csv'];

trainTbl = readtable(sTrainingPath);

trainPaths = trainTbl.Path;
trainWidths = trainTbl.Width;
trainHeights = trainTbl.Height;
trainRoiX1 = trainTbl.Roi_X1;
trainRoiY1 = trainTbl.Roi_Y1;
trainRoiX2 = trainTbl.Roi_X2;
trainRoiY2 = trainTbl.Roi_Y2;
trainClasses = trainTbl.ClassId;

testTbl = readtable(sTestPath);

testPaths = testTbl.Path;
testWidths = testTbl.Width;
testHeights = testTbl.Height;
testRoiX1 = testTbl.Roi_X1;
testRoiY1 = testTbl.Roi_Y1;
testRoiX2 = testTbl.Roi_X2;
testRoiY2 = testTbl.Roi_Y2;
testClasses = testTbl.ClassId;

 
 A = zeros(length(testPaths), 50*50);
 
 for i = 1:length(testPaths)
 
     % Read in the image at the path
     RGB = imread([sBasePath, char(testPaths(i))]);
 
     % Perhaps grab the region of interest
     
     if (testRoiY1(i) == 0)
         testRoiY1(i) = 1;
     end
     if (testRoiY2(i) == 0)
         testRoiY2(i) = 1;
     end
     if (testRoiX1(i) == 0)
         testRoiX1(i) = 1;
     end
     if (trainRoiX2(i) == 0)
         testRoiX2(i) = 1;
     end
     RGB_cropped = RGB(testRoiY1(i):testRoiY2(i), testRoiX1(i):testRoiX2(i), :);
 
     % Resize the image to an experimentally-determined size
     %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
     RGB_rescaled = imresize(RGB, [50 50]);
     
     GRAY = rgb2gray(RGB_rescaled);
     
     A(i, :) = reshape(GRAY, 1, 50*50);
 end

save('signstest.mat', 'A');

A = zeros(length(trainPaths), 50*50);
 
for i = 1:length(trainPaths)
 
     % Read in the image at the path
     RGB = imread([sBasePath, char(trainPaths(i))]);
 
     % Perhaps grab the region of interest
     
     if (trainRoiY1(i) == 0)
         trainRoiY1(i) = 1;
     end
     if (trainRoiY2(i) == 0)
         trainRoiY2(i) = 1;
     end
     if (trainRoiX1(i) == 0)
         trainRoiX1(i) = 1;
     end
     if (trainRoiX2(i) == 0)
         trainRoiX2(i) = 1;
     end
     RGB_cropped = RGB(trainRoiY1(i):trainRoiY2(i), trainRoiX1(i):trainRoiX2(i), :);
 
     % Resize the image to an experimentally-determined size
     %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
     RGB_rescaled = imresize(RGB, [50 50]);
     
     GRAY = rgb2gray(RGB_rescaled);
     
     A(i, :) = reshape(GRAY, 1, 50*50);
end

save('signstrain.mat', 'A');

%signstrain = load('C:\Users\j39950\Documents\MATLAB\670 Machine Learning\signstrain.mat');
%signstest  = load('C:\Users\j39950\Documents\MATLAB\670 Machine Learning\signstest.mat');

signstrain = load('signstrain.mat');
signstest  = load('signstest.mat');

% Training data
A = signstrain.A;
A_labels = trainClasses;

% Test data
test_data = signstest.A;

% Image dimensions
M = 50;
N = 50;
num_pixels = M*N;

num_signs = 43;

%% 2. Naive Bayes on first 100 eigenvalues of PCA

% m - mean: 1xc matrix, jth column is mean of the jth class
% S - 1x1xc matrix, covariance matrix of the normal distribution of the jth
% class
% P - c-dimensional vector, a priori probability of jth class
% X - 1xN matrix, columns are the data vectors to be classified

num_training_data = size(A_labels,1);
num_features = 100;  % switch around

P = zeros(1,num_signs);
S = zeros(num_features,num_features,num_signs);
m = zeros(num_features,num_signs);
for i = 1:num_signs
    % P: c-dimensional vector, whose j-th component is the a priori
    % probability of the j-th class.
    P(i) = sum(A_labels == i)/num_training_data; 
    % m - mean: 1xc matrix, jth column is mean of the jth class
    m(:,i) = mean(A(A_labels'==i,1:num_features))';
    S(:,:,i) = cov(A(A_labels'==i,1:num_features));
end

test_label_zb=bayes_classifier(m,S,P,test_data(:,1:num_features)');

%% Compuare with testClasses

