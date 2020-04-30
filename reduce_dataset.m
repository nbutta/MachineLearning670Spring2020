%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% This function reads in an image dataset and determines the images with
% the best contrast (low variance of image histogram) and highest
% resolution. This higher quality data can be used to train our models and
% also for selecting which test images to predict.

% We first get the best resolution images of each class using resTrim which
% is a percentage of images of each class to take with the best resolution

% We then get the best contrast images of each class using contrastTrim 
% which is a percentage of images of each class to take with the best
% contrast

% Final number of images = (Original num. images * resTrim) * contrastTrim)

% This comes per the grader's recommendations:
% "I recommend eliminating the low image quality data (low contrast and 
% poor resolution) and get high performance classification results.  
% If the schedule permits, the low-quality data can be introduced to assess
% the robustness of the algorithm and the degradation of the algorithm 
% performance under various conditions.  

%%
function [paths, widths, heights, roiX1, roiY1, roiX2, roiY2, classes] = reduce_dataset(basePath, csvPath, resTrim, contrastTrim)

    tbl = readtable([basePath, csvPath]);

    paths = tbl.Path;
    widths = tbl.Width;
    heights = tbl.Height;
    roiX1 = tbl.Roi_X1;
    roiY1 = tbl.Roi_Y1;
    roiX2 = tbl.Roi_X2;
    roiY2 = tbl.Roi_Y2;
    classes = tbl.ClassId;
    
    mean(widths.*heights)

    % Sort all the data by class ID (ascending)
    [classes, I] = sort(classes);
    paths = paths(I);
    widths = widths(I);
    heights = heights(I);
    roiX1 = roiX1(I);
    roiY1 = roiY1(I);
    roiX2 = roiX2(I);
    roiY2 = roiY2(I);

    num_uniq_signs = 43;

    figure
    histogram(classes);

    %% Resolution

    best_res_ind = [];

    img_res = widths.*heights;

    for i = 1:num_uniq_signs

        % Get the resolution for each image in the class
        class_res = img_res(classes==i-1);
        og_ind = find(classes==i-1);

        % Sort the resolutions (ascending)
        [class_res, I_res] = sort(class_res, 'descend');

        high_res_ind = og_ind(I_res(1:round(resTrim*end)));

        %high_res_classes = classes(high_res_ind);

        best_res_ind = [best_res_ind; high_res_ind];

    end

    % figure
    % histogram(high_res_classes)
    % xlabel('Class ID')
    % ylabel('Number of high resolution images')
    % title('Number of high res images per class')

    %% Update for only high res images

    num_high_res_signs = length(best_res_ind);

    classes = classes(best_res_ind);
    paths = paths(best_res_ind);
    widths = widths(best_res_ind);
    heights = heights(best_res_ind);
    roiX1 = roiX1(best_res_ind);
    roiY1 = roiY1(best_res_ind);
    roiX2 = roiX2(best_res_ind);
    roiY2 = roiY2(best_res_ind);

    figure;
    histogram(classes);
    
    mean(widths.*heights)

    %% Contrast

    best_var_ind = [];

    intensity_var = zeros(num_high_res_signs, 1);

    for i = 1:num_high_res_signs

        % Read in the image at the path
        RGB = imread([basePath, char(paths(i))]);

        % Resize to common size
        RGB = imresize(RGB, [50 50]);

        % Compute the intensity histogram
        [COUNTS,X] = imhist(rgb2gray(RGB));

        % Determine the variance of the intensities
        intensity_var(i) = var(COUNTS);

    end
    
    mean(intensity_var)

    for i = 1:num_uniq_signs

        % Get the intensity variance for each image in the class
        class_var = intensity_var(classes==i-1);
        og_ind = find(classes==i-1);

        % Sort the variances (descending)
        [class_var, I_var] = sort(class_var);

        high_var_ind = og_ind(I_var(1:round(contrastTrim*end)));

        %high_res_classes = classes(high_res_ind);

        best_var_ind = [best_var_ind; high_var_ind];

    end

    num_high_qual_signs = length(best_var_ind);

    classes = classes(best_var_ind);
    paths = paths(best_var_ind);
    widths = widths(best_var_ind);
    heights = heights(best_var_ind);
    roiX1 = roiX1(best_var_ind);
    roiY1 = roiY1(best_var_ind);
    roiX2 = roiX2(best_var_ind);
    roiY2 = roiY2(best_var_ind);

    figure;
    histogram(classes);
    
    mean(intensity_var(best_var_ind))

    % % Sort the intensity variance (ascending)
    % [intensity_var, I_var] = sort(intensity_var);
    % 
    % % Keep track of how many images of each type have high contrast
    % high_contrast = zeros(1, num_uniq_signs);
    % 
    % high_contrast_classes = classes(I_var(1:round(.75*end)));
    % 
    % figure;
    % histogram(high_contrast_classes)
    % xlabel('Class ID')
    % ylabel('Number of high contrast images')
    % title('Number of high contrast images per class')

    %% Examples

%     figure;
%     RGB = imread([sBasePath, char(paths(I(round(.75*end))))]);
%     RGB = imresize(RGB, [50 50]);
%     subplot(121)
%     imshow(RGB)
%     title('contrast')
%     subplot(122)
%     imhist(rgb2gray(RGB))
%     title('intensity histogram')

    % figure;
    % RGB = imread([sBasePath, char(paths(I(end)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('bad contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure;
    % RGB = imread([sBasePath, char(paths(I(end-1)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('bad contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure;
    % RGB = imread([sBasePath, char(paths(I(end-2)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('bad contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure
    % RGB = imread([sBasePath, char(paths(I(round(end/2))))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('medium contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure
    % RGB = imread([sBasePath, char(paths(I(1)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('good contrast');
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure
    % RGB = imread([sBasePath, char(paths(I(2)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('good contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')
    % 
    % figure
    % RGB = imread([sBasePath, char(paths(I(3)))]);
    % RGB = imresize(RGB, [50 50]);
    % subplot(121)
    % imshow(RGB)
    % title('good contrast')
    % subplot(122)
    % imhist(rgb2gray(RGB))
    % title('intensity histogram')

end

