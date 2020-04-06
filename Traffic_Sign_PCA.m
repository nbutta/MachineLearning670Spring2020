%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

%% PCA-Based Traffic Sign Recognition

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


A = signstrain.A;
A_labels = trainClasses;

% Image dimensions
M = 50;
N = 50;
num_pixels = M*N;

num_signs = 43;

%% 2. PCA Feature Extraction

% Using all the training photographs for the people in the training dataset,
% construct a subspace with dimensionality H less than or equal to such 
% that this subspace has the maximum dispersion for the projections. 
% To extract this subspace, use Principal Component Analysis

[eigsigns, eigvals] = pca_basis(A);

% Produce a column vector of eigenvalues
eigvals = diag(eigvals);

%% 3. Plot the cumulative eigenvalues (cumulative variance)

% figure;
% subplot(1, 3, 1)
% bar(eigvals./sum(eigvals))
% %title('variance described by individual eigenvectors')
% xlabel('eigenvector')
% ylabel('% of variance')
% 
% subplot(1, 3, 2)
% semilogy(diag(eigvals), 'ko', 'LineWidth', [2])
% %title('eigenvalues of individual eigenvectors')
% xlabel('eigenvector')
% ylabel('eigenvalue')
% 
% subplot(1, 3, 3)
% stairs([0; cumsum(eigvals./sum(eigvals))])
% %title('cumulative variance described by eigenvectors')
% xlabel('eigenvector')
% ylabel('cumulative variance')

%% 4. Plot the first 3 eigensigns and the last eigensign

% Observations: The eigen signs themselves form a basis set of images used
% to construct the covariance matrix. Any sign in the set of images can be
% considered as a combination of these "standard" signs. The first three
% eigensigns together represent 77.2% of the variance in the data. The first
% eigensign represents 56.6% of the variance alone. With 40 eigensigns, we can 
% represent basically 100% of the total variance. This means we can get a
% fair approximation of most signs with only the first three eigensigns.
% The eigensigns created in this case appear as light and dark areas
% arranged in the rough shape of a sign. In the first eigensign, 
% you can see the outlines of a triangluar and circular sign and can make
% out some text digits as well. This sign resembles a light sign with a dark background
% The second sign appears as almost the
% inverse of the first eigensign to account for darker signs with lighter
% backgrounds. The last eigen sign apprears as a combination of all
% different types of signs with both an normal and inverted triangle.
% If we consider any image as a combination of these
% eigensigns, then it makes sense that we can start with the basis of the first
% three eigensigns and then use combinations of the subsequent eigensigns
% to correct and fine-tune the basic images with more specific features
% expressed using later eigensigns that generally describe less variance.

% figure;
% subplot(1, 4, 1), sign1 = reshape(eigsigns(:,1), M, N); pcolor(flipud(sign1)), shading interp, colormap(gray)
% title('first eigensign')
% subplot(1, 4, 2), sign2 = reshape(eigsigns(:,2), M, N); pcolor(flipud(sign2)), shading interp, colormap(gray)
% title('second eigensign')
% subplot(1, 4, 3), sign3 = reshape(eigsigns(:,3), M, N); pcolor(flipud(sign3)), shading interp, colormap(gray)
% title('third eigensign')
% subplot(1, 4, 4), sign4 = reshape(eigsigns(:,end), M, N); pcolor(flipud(sign4)), shading interp, colormap(gray)
% title('last eigensign')

% Could alternatively use imagesc to display signs, as follows

% figure;
% subplot(1, 4, 1), sign1 = reshape(eigsigns(:,1), M, N); imagesc(sign1), colormap(gray)
% title('first eigensign')
% subplot(1, 4, 2), sign2 = reshape(eigsigns(:,2), M, N); imagesc(sign2), colormap(gray)
% title('second eigensign')
% subplot(1, 4, 3), sign3 = reshape(eigsigns(:,3), M, N); imagesc(sign3), colormap(gray)
% title('third eigensign')
% subplot(1, 4, 4), sign4 = reshape(eigsigns(:,end), M, N); imagesc(sign4), colormap(gray)
% title('last eigensign')

%% 5. Pick a sign and reconstruct it using k = 10, 20, 30, 40 eigenvectors. 

% Plot all of these reconstructions and compare them. 
% For each value of k, plot the original image, reconstructed image, 
% and the difference b/w the original image and reconstruction in each case. 
% Write your observations and an explanation.

% Observations:
% You can run this section of code multiple times with different signs.
% As we include more and more eigenvectors, the reconstructed sign begins
% to look more and more like the original sign image, as expected. More of
% the unique features start to become visible, like glasses or hair or
% mouth and nose. However, it's very difficult for the naked eye to be able
% to say who the person is without the original image available.

% When looking at the image differences, you will see black areas where the
% original images are similar and white areas where they are different.
% Again, as expected, the amount of white generally appears to decrease as
% more eigenvectors are used for reconstruction.

% Pick a random sign
% r = randi([1 size(A,1)]);
% chosen_sign = A(r,:);
% 
% recon10 = reshape(pca_reconstruction(chosen_sign, eigsigns, 10), M, N);
% recon20 = reshape(pca_reconstruction(chosen_sign, eigsigns, 20), M, N);
% recon30 = reshape(pca_reconstruction(chosen_sign, eigsigns, 30), M, N);
% recon40 = reshape(pca_reconstruction(chosen_sign, eigsigns, 40), M, N);
% 
% % Reshape the chosen sign
% chosen_sign = reshape(chosen_sign, M, N);
% 
% % Plot the original sign in the first column
% figure;
% subplot(4, 3, 1), pcolor(flipud(chosen_sign)), shading interp, colormap(gray), title(strcat('original sign: ', num2str(A_labels(r))))
% subplot(4, 3, 4), pcolor(flipud(chosen_sign)), shading interp, colormap(gray), title(strcat('original sign: ', num2str(A_labels(r))))
% subplot(4, 3, 7), pcolor(flipud(chosen_sign)), shading interp, colormap(gray), title(strcat('original sign: ', num2str(A_labels(r))))
% subplot(4, 3, 10), pcolor(flipud(chosen_sign)), shading interp, colormap(gray), title(strcat('original sign: ', num2str(A_labels(r))))
% 
% % Plot the reconstructed signs with k = 10, 20, 30, 40 eigenvectors
% subplot(4, 3, 2), pcolor(flipud(recon10)), shading interp, colormap(gray), title ('k = 10 reconstruction')
% subplot(4, 3, 5), pcolor(flipud(recon20)), shading interp, colormap(gray), title ('k = 20 reconstruction')
% subplot(4, 3, 8), pcolor(flipud(recon30)), shading interp, colormap(gray), title ('k = 30 reconstruction')
% subplot(4, 3, 11), pcolor(flipud(recon40)), shading interp, colormap(gray), title ('k = 40 reconstruction')
% 
% % Scale the reconstructions to 0-255 like the the input image
% s10 = single(255*(mat2gray(recon10)));
% s20 = single(255*(mat2gray(recon20)));
% s30 = single(255*(mat2gray(recon30)));
% s40 = single(255*(mat2gray(recon40)));
% 
% % Plot image differences. 
% % Black = small difference, White = high difference
% subplot(4, 3, 3),
% imagesc(abs(s10-chosen_sign)), colormap(gray), title('image difference')
% subplot(4, 3, 6),
% imagesc(abs(s20-chosen_sign)), colormap(gray), title('image difference')
% subplot(4, 3, 9),
% imagesc(abs(s30-chosen_sign)), colormap(gray), title('image difference')
% subplot(4, 3, 12),
% imagesc(abs(s40-chosen_sign)), colormap(gray), title('image difference')

% Alternate way of showing images differences

% subplot(4, 3, 3),
% imshowpair(chosen_sign,recon10,'diff')
% subplot(4, 3, 6),
% imshowpair(chosen_sign,recon20,'diff')
% subplot(4, 3, 9),
% imshowpair(chosen_sign,recon30,'diff')
% subplot(4, 3, 12),
% imshowpair(chosen_sign,recon40,'diff')

% sum(abs(s10-chosen_sign), 'all')
% sum(abs(s20-chosen_sign), 'all')
% sum(abs(s30-chosen_sign), 'all')
% sum(abs(s40-chosen_sign), 'all')

%% 6. Load the test data

% test_signs = load('C:\Users\j39950\Documents\MATLAB\670 Machine Learning\test.mat');
% test_data = test_signs.test_data;
test_data = signstest.A;

%% 7. Classify each of the test images
%  8. Show the closest images in the training set for each test

% Determine the projection of each test photo onto H 
% with different dimensionalities d = 10, 20, 30, 40

% Compare the distance of this projection to the projections of all 
% images in the training data.

% For each test photo's projection, find the closest category of projection
% in the training data.

num_test_signs = length(testClasses);

known_classes = testClasses(1:num_test_signs);
predicted_classes = zeros(1, num_test_signs);
training_scores = A*eigsigns;

%for i = 1:length(testClasses)
for i = 1:num_test_signs 
   % figure    
        
    test_sign = test_data(i,:);
    tf = reshape(test_sign, M, N);
    
    % Plot the test sign
%     subplot(4, 3, 1), pcolor(flipud(tf)), shading interp, colormap(gray), title(strcat('test sign: ', num2str(testClasses(i))))
%     subplot(4, 3, 4), pcolor(flipud(tf)), shading interp, colormap(gray), title(strcat('test sign: ', num2str(testClasses(i))))
%     subplot(4, 3, 7), pcolor(flipud(tf)), shading interp, colormap(gray), title(strcat('test sign: ', num2str(testClasses(i))))
%     subplot(4, 3, 10), pcolor(flipud(tf)), shading interp, colormap(gray), title(strcat('test sign: ', num2str(testClasses(i))))
    
    % Project the test sign onto the k = 10, 20, 30, 40 eigenvectors
    % Plot the reconstructions of the test sign using k = 10, 20, 30 and
    % 40 eigensigns    
%     tst_recon10 = reshape(pca_reconstruction(test_sign, eigsigns, 10), M, N);
%     tst_recon20 = reshape(pca_reconstruction(test_sign, eigsigns, 20), M, N);
%     tst_recon30 = reshape(pca_reconstruction(test_sign, eigsigns, 30), M, N);
%     tst_recon40 = reshape(pca_reconstruction(test_sign, eigsigns, 40), M, N);
    
%     subplot(4, 3, 2), pcolor(flipud(tst_recon10)), shading interp, colormap(gray), title ('k = 10 reconstruction')
%     subplot(4, 3, 5), pcolor(flipud(tst_recon20)), shading interp, colormap(gray), title ('k = 20 reconstruction')
%     subplot(4, 3, 8), pcolor(flipud(tst_recon30)), shading interp, colormap(gray), title ('k = 30 reconstruction')
%     subplot(4, 3, 11), pcolor(flipud(tst_recon40)), shading interp, colormap(gray), title ('k = 40 reconstruction')
     
    % Find best match in the training data using 10 eigenvectors  
%     [pred, closest_sign] = pca_predict(test_sign, eigsigns, 10, A, A_labels);
%     subplot(4, 3, 3), cf = reshape(closest_sign, M, N); pcolor(flipud(cf)), shading interp, colormap(gray)
%     title(strcat('best match: ', num2str(pred)))
%     
%     % Find best match in the training data using 20 eigenvectors
%     [pred, closest_sign] = pca_predict(test_sign, eigsigns, 20, A, A_labels);
%     subplot(4, 3, 6), cf = reshape(closest_sign, M, N); pcolor(flipud(cf)), shading interp, colormap(gray)
%     title(strcat('best match: ', num2str(pred)))
%     
%     % Find best match in the training data using 30 eigenvectors
%     [pred, closest_sign] = pca_predict(test_sign, eigsigns, 30, A, A_labels);
%     subplot(4, 3, 9), cf = reshape(closest_sign, M, N); pcolor(flipud(cf)), shading interp, colormap(gray)
%     title(strcat('best match: ', num2str(pred)))
    
    % Find best match in the training data using 40 eigenvectors
    
   % training_scores = A*eigsigns;
  %  [pred, closest_sign] = pca_predict(test_sign, eigsigns, 40, A, A_labels);
    [pred2, closest_sign2] = pca_predict2(test_sign, eigsigns, 40, training_scores, A, A_labels);
%     subplot(4, 3, 12), cf = reshape(closest_sign, M, N); pcolor(flipud(cf)), shading interp, colormap(gray)
%     title(strcat('best match: ', num2str(pred)))
       
    predicted_classes(i) = pred2;
end

% (https://www.mathworks.com/help/deeplearning/ref/confusionchart.html)

fig = figure;
[C, order] = confusionmat(known_classes, predicted_classes);
cm = confusionchart(C, 'RowSummary','row-normalized','ColumnSummary','column-normalized');

% This will sort based on the true positive rate
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';