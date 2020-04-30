%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

%% Traffic Sign Recognition

clear all   % clear workspace
close all   % close all figure windows
clc         % Comannd Line Clear

%% Load the meta csv

sBasePath = 'C:/Users/j39950/Documents/JHU/Courses/2020/670 Machine Learning/Proj/Traffic Sign Recognition/Kaggle/'; 

sMetaPath = [sBasePath, 'Meta.csv'];
fID = fopen([sBasePath, 'Meta.csv'], 'r');

metaTbl = readtable(sMetaPath);

metaPaths = metaTbl.Path;
metaClasses = metaTbl.ClassId;

[metaClasses, I] = sort(metaClasses);
metaPaths = metaPaths(I);

num_signs = length(metaPaths);

%A = zeros(100, 100, 3, length(metaPaths));

HH = zeros(num_signs, 100);

for i = 1:num_signs

    % Read in the image at the path
    RGB = imread([sBasePath, char(metaPaths(i))]);
    
    [hog1,visualization] = extractHOGFeatures(RGB,'CellSize',[32 32]);
    [featureVector,hogVisualization] = extractHOGFeatures(RGB);
    
    % Resize the image to an experimentally-determined size
    %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
    dims = size(RGB);
    
%     if (dims(1) ~= 100 || dims(2) ~= 100) 
%         RGB_rescaled = imresize(RGB, [100 100]);
%     else
%         RGB_rescaled = RGB;
%     end

    %Display the original image and the HOG features.

    figure
    subplot(1,3,1);
    imshow(RGB);
    hold on;
    plot(hogVisualization);
    subplot(1,3,2);
    plot(visualization);
    subplot(1,3,3);
    histogram(featureVector, 100);
    
    
    GRAY = rgb2gray(RGB);
    % Identifying round objects for detecting round signs
    %threshold = graythresh(GRAY);
    BW = imbinarize(GRAY);
    notBW = ~BW;

    figure
    subplot(3, 6, 7), imshow(BW)
    BW = bwareaopen(BW,100);
    subplot(3, 6, 8), imshow(BW)
    se = strel('disk',2);
    BW = imclose(BW,se);
    subplot(3, 6, 9), imshow(BW)
    BW = imfill(BW,'holes');
    subplot(3, 6, 10), imshow(BW)
    [B,L] = bwboundaries(BW, 'noholes');
    subplot(3, 6, 11), imshow(BW)

%     subplot(3, 6, 13), imshow(notBW)
%     notBW = bwareaopen(notBW,100);
%     subplot(3, 6, 14), imshow(notBW)
%     se = strel('disk',2);
%     notBW = imclose(notBW,se);
%     subplot(3, 6, 15), imshow(notBW)
%     notBW = imfill(~notBW,'holes');
%     subplot(3, 6, 16), imshow(notBW)
%     [notB, notL] = bwboundaries(notBW, 'noholes');
%     subplot(3, 6, 17), imshow(notBW)
    BW = imresize(BW, [100 100]);
    BW_new = zeros(120, 120);
    BW_new(11:110, 11:110) = BW;
    BW = BW_new;

    test = zeros(120, 120);
    test(20:80, 20:100) = ones(length(20:80), length(20:100)); 

    BW = test;
    
    figure
    imshow(BW, []);
    title('Cleaned Binary Image')
    hold on
    

    
    [labeledImage, numberOfObjcts] = bwlabel(BW);
    blobMeasurements = regionprops(labeledImage,'Perimeter','Area', 'Centroid'); 
    % for square ((a>17) && (a<20))
    % for circle ((a>13) && (a<17))
    % for triangle ((a>20) && (a<30))
    circularities = [blobMeasurements.Perimeter].^2 ./ (4 * pi * [blobMeasurements.Area]);
    hold on;
    % Say what they are
    for blobNumber = 1 : numberOfObjcts
      if circularities(blobNumber) < 1.19
        message = sprintf('The circularity of object #%d is %.3f, so the object is a circle',...
          blobNumber, circularities(blobNumber));
        theLabel = 'Circle';
      elseif circularities(blobNumber) < 1.40
        message = sprintf('The circularity of object #%d is %.3f, so the object is a Rectangle',...
          blobNumber, circularities(blobNumber));
        theLabel = 'Rectangle';
      else
        message = sprintf('The circularity of object #%d is %.3f, so the object is a triangle',...
          blobNumber, circularities(blobNumber));
        theLabel = 'Triangle';
      end
      text(blobMeasurements(blobNumber).Centroid(1), blobMeasurements(blobNumber).Centroid(2),...
        theLabel, 'Color', 'r');
      uiwait(msgbox(message));
    end
    
end



