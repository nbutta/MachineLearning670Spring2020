%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% Function to compute the hue histogram of an image dataset

% The number of features is the number of histogram bins equally spaced
% between 0 and 1. 

% HH is a matrix where the histogram bin counts are in the columns
% Each row of HH is a new image

function HH = get_hue_histograms(basePath, paths, roiX1, roiY1, roiX2, roiY2, num_features)

    num_signs = length(paths);

    HH = zeros(num_signs, num_features);

    for i = 1:num_signs

        % Read in the image at the path
        RGB = imread([basePath, char(paths(i))]);

        RGB_roi = RGB(roiY1(i):roiY2(i), roiX1(i):roiX2(i), :);
        
        % Resize the image to an experimentally-determined size
        RGB_rescaled = imresize(RGB_roi, [50 50]);

        % Convert to HSV color space
        HSV = rgb2hsv(RGB_rescaled);

        [N, edges] = histcounts(HSV(:,:,1), 0:1/num_features:1);
        HH(i, :) = N;
        
        % Compute HSV Histogarm
        %figure
        %subplot(121)
        %hue_hist = histogram(HSV(:,:,1), 100);

        %title(char(paths(i)))

        %subplot(122)
        %imshow(RGB_rescaled)

        %h = hue_hist.Values;
        %HH(i, :) = h;

    end
end