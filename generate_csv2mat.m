%%
% File: generate_data.m
%   Reads the input csv file (Train.csv or Test.csv), pre-process the
%   read images and save the output to a matrix.
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%   Term: Spring 2020
%

function [data] = generate_csv2mat(fInputName,fOutName,varargin)

p = inputParser;

p.KeepUnmatched = true;
p.addOptional('imPreProcess',false);  

p.parse(p,varargin{:});
imPreProcess = p.Results.imPreProcess;

data = [];

sBasePath = fullfile(fileparts(fullfile(mfilename('fullpath'))),'..','gtsrb-german-traffic-sign/');

dataTbl = readtable(fInputName);

paths = dataTbl.Path;
widths = dataTbl.Width;
heights = dataTbl.Height;
roiX1 = dataTbl.Roi_X1;
roiY1 = dataTbl.Roi_Y1;
roiX2 = dataTbl.Roi_X2;
roiY2 = dataTbl.Roi_Y2;
classes = dataTbl.ClassId;

 
A = zeros(length(paths), 50*50);
B = zeros(length(paths), 50*50);
 
for i = 1:length(paths)
 
     % Read in the image at the path
     RGB = imread([sBasePath, char(paths(i))]);
 
     % Perhaps grab the region of interest
     
     if (roiY1(i) == 0)
         roiY1(i) = 1;
     end
     if (roiY2(i) == 0)
         roiY2(i) = 1;
     end
     if (roiX1(i) == 0)
         roiX1(i) = 1;
     end
     if (roiX2(i) == 0)
         roiX2(i) = 1;
     end
     RGB_cropped = RGB(roiY1(i):roiY2(i), roiX1(i):roiX2(i), :);
 
     % Resize the image to an experimentally-determined size
     %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
     RGB_rescaled = imresize(RGB, [50 50]);
     
     GRAY = rgb2gray(RGB_rescaled);
     
     A(i, :) = reshape(GRAY, 1, 50*50);
     
     if imPreProcess == true
         J = histeq(GRAY);
         B(i,:) = reshape(J, 1, 50*50);
     end
end

data.A = A;
data.classes = classes;
data.paths = paths;
paths = dataTbl.Path;
data.widths = widths;
data.heights = heights;
data.roiX1 = roiX1;
data.roiY1 = roiY1;
data.roiX2 = roiX2;
data.roiY2 = roiY2;

if imPreProcess == true
    data.B = B;
    save(fOutName, 'A','B',...
    'classes','paths','widths','heights',...
    'roiX1','roiY1','roiX2','roiY2');
else
      save(fOutName, 'A',...
    'classes','paths','widths','heights',...
    'roiX1','roiY1','roiX2','roiY2');  
end

end