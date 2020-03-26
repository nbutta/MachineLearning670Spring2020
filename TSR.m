%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% Steps to complete:

% 1. Loading our training and testing split from the GTSRB dataset
% 2. Preprocessing the images
% 3. Training our model
% 4. Evaluating our model’s accuracy
% 5. Serializing the model to disk so we can later use it to make 
%    predictions on new traffic sign data

%% Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% Load the training csv

sBasePath = 'C:/Users/j39950/Documents/JHU/Courses/2020/670 Machine Learning/Proj/Traffic Sign Recognition/Kaggle/'; 

sTrainingPath = [sBasePath, 'Train.csv'];
fID = fopen([sBasePath, 'Train.csv'], 'r');

tbl = readtable(sTrainingPath);

%% Preprocess the data

% output = preprocess_data(sBasePath, tbl);

%% Feature Extraction

% output = extract_features();

trainPaths = tbl.Path;
trainRoiX1 = tbl.Roi_X1;
trainRoiY1 = tbl.Roi_Y1;
trainRoiX2 = tbl.Roi_X2;
trainRoiY2 = tbl.Roi_Y2;
%i = 38549;
i = randi([1 length(trainPaths)]);
RGB = imread([sBasePath, char(trainPaths(i))]);


%%  Detect roundness...

RGB_cropped = RGB(trainRoiY1(i):trainRoiY2(i), trainRoiX1(i):trainRoiX2(i), :);
montage({RGB, RGB_cropped});
RGB_rescaled = imresize(RGB, [50 50]);
figure, subplot(3, 5, 1), imshow(RGB_rescaled)

output = detect_roundness(RGB_rescaled);


%% KMeans the image

output = do_kmeans(RGB);


%% Apply PCA

% look at Traffic_Sign_PCA script