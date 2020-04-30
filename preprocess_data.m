% In order to successfully train an accurate traffic sign classifier 
% we’ll need to devise an experiment that can:
% 1. Preprocess our input images to improve contrast.
% 2. Account for class label skew
% 3. Make all of our image the same size

% We could account for class label skew in a few ways:
% 1. Generate new images for classes with fewer instances by applying 
%    an image augmenter or adding random noise to existing images
% 2. Ignore (delete) images from classes with a number of instances above
%    the class with the least number of instances.

% Class 0, 19, 32, 37 have the least number of images (210)

% In order to improve image contrast, we can do some type of histogarm
% equalization

function output = preprocess_data(sBasePath, tbl)

    output = [];

    trainPaths = tbl.Path;
    trainWidths = tbl.Width;
    trainHeights = tbl.Height;
    trainRoiX1 = tbl.Roi_X1;
    trainRoiY1 = tbl.Roi_Y1;
    trainRoiX2 = tbl.Roi_X2;
    trainRoiY2 = tbl.Roi_Y2;
    trainClasses = tbl.ClassId;
    
    % Should be 43 total classes (0-42)
    numClasses = length(unique(trainClasses));

    % classHist = zeros(1, numClasses);
    % for i = 1:numClasses
    %     classHist(i) = sum(trainClasses==i-1);    
    % end

%     [classCounts, classValues] = hist(trainClasses, unique(trainClasses));
%     bar(classValues, classCounts);
%     xlabel('Class')
%     ylabel('Occurences')
%     title('Kaggle GTSRB Class Distribution')
% 
%     % Averages are 50x50 image basically. This would be good to use as 
%     % the image rescaling criteria.
%     avg_height = mean(trainHeights)
%     avg_width = mean(trainWidths)
%     avg_aspect_ratio = avg_width/avg_height
%     
%     avg_heights = zeros(43, 1);
%     std_heights = zeros(43, 1);
%     avg_widths = zeros(43, 1);
%     std_widths = zeros(43, 1);
%     for i = 1:43
%        idx = trainClasses==i-1;
%        avg_widths(i) = mean(trainWidths(idx));
%        std_widths(i) = std(trainWidths(idx));
%        avg_heights(i) = mean(trainHeights(idx));
%        std_heights(i) = std(trainHeights(idx));
%     end
%     
%     bar(classValues, avg_heights);
%     title('Average image heights by class');
%     xlabel('Class')
%     ylabel('Average height (# pixels)')
%     figure
%     barh(classValues, avg_widths);
%     title('Average image widths by class');
%     xlabel('Average width (# pixels)')
%     ylabel('Class')
%     
%     figure
%     boxplot(trainWidths, trainClasses)
%     
%     figure
%     boxplot(trainHeights, trainClasses)
%     
%     figure
%     errorbar(avg_heights, std_heights, 'x');
%     title('Average image height & std by class');
%     xlabel('Class')
%     ylabel('Height (# pixels)')
%     
%     figure
%     errorbar(avg_widths, std_widths, 'x');
%     title('Average image width & std by class');
%     xlabel('Class')
%     ylabel('Width (# pixels)')

    % Example of an image augmenter declaration, which will randomly rotate
    % and translate image pixels
    
%     imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXShear',[-20,20], ...
%     'RandYShear',[-20,20], ...
%     'RandXTranslation',[-5 5], ...
%     'RandYTranslation',[-5 5])

%     imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXShear',[-20,20], ...
%     'RandYShear',[-20,20])
% 
% %     In the end, we will have to process all images using something like
% %     this:
%     
%     %for i = 1:length(trainPaths)
%     for i = 1:10
%         
%         idx = randi([1 length(trainClasses)]);
%         % Read in the image at the path
%         RGB = imread([sBasePath, char(trainPaths(idx))]);
%         
%         % Perhaps grab the region of interest
%         RGB_cropped = RGB(trainRoiY1(idx):trainRoiY2(idx), trainRoiX1(idx):trainRoiX2(idx), :);
%         
%         % Resize the image to an experimentally-determined size
%         %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
%         RGB_rescaled = imresize(RGB_cropped, [50 50]);
%         figure
%         subplot(411)
%         montage({RGB, RGB_cropped})
%         title('Original vs ROI cropped')
%         
%         subplot(412)
%         montage({RGB_cropped, RGB_rescaled})
%         title('ROI cropped vs Rescaled 50x50')
%         
%         % Process the image and produce an output
%         
%         % Do image augmentation to increase the amount of training data
%         % https://www.mathworks.com/help/deeplearning/ref/imagedataaugmenter.html
%         % https://www.mathworks.com/help/deeplearning/ug/preprocess-images-for-deep-learning.html
%         
%         % For example: 
%         
%         % Rescale to 0-1
%         r = rescale(RGB_rescaled(:,:,1));
%         g = rescale(RGB_rescaled(:,:,2));
%         b = rescale(RGB_rescaled(:,:,3));
% 
%         % Add random gaussian noise to the rescaled image
%         r_noisy = imnoise(r, 'gaussian');
%         g_noisy = imnoise(g, 'gaussian');
%         b_noisy = imnoise(b, 'gaussian');
% 
%         % Put the RGB image with noise back together
%         RGB_noisy = cat(3, r_noisy, g_noisy, b_noisy);
% 
%         % Augment the image and show the side-by-side with the original
%         augmentedImage = augment(imageAugmenter, RGB_rescaled);
%         
%         subplot(413)
%         montage({RGB_rescaled, RGB_noisy})
%         title('Rescaled vs Noisy image')
%         
%         subplot(414)
%         montage({RGB_rescaled, augmentedImage})
%         title('Rescaled vs Augmented image')
%         
%         % Do histogram equalization
%         % (https://www.mathworks.com/help/images/adaptive-histogram-equalization.html)
%         
%     end

    % For now, lets just see what happens when we do these things on a
    % single image so we can follow along:
    i = 38549;
    %i = 7217;
    %i = randi([1 length(trainClasses)])
    RGB = imread([sBasePath, char(trainPaths(i))]);
    
    % Perhaps grab the region of interest. Not 
    RGB_cropped = RGB(trainRoiY1(i):trainRoiY2(i), trainRoiX1(i):trainRoiX2(i), :);

    % Resize the image to an experimentally-determined size
    %(https://www.mathworks.com/help/images/ref/imresize.html#d120e151526)
    RGB_rescaled = imresize(RGB, [50 50]);
    
    % or
    % RBG_rescaled = imresize(RGB_cropped, [50 50])
    
    figure
    montage({RGB, RGB_rescaled})
    
    GRAY_rescaled = rgb2gray(RGB_rescaled);

    figure
    montage({RGB_rescaled, GRAY_rescaled})

    % Following this procedure for colored images:
    % (https://www.mathworks.com/help/images/ref/adapthisteq.html)
    
    % Testing different clip limits (higher limits = more contrast)
    % The optimal number of tiles depends on the type of the input image, 
    % and it is best determined through experimentation
    
    tileSize = [4 4];
    clipLimit = .000025;
    multi = {RGB_rescaled};
    LAB1 = rgb2lab(RGB_rescaled);
    L = LAB1(:,:,1)/100;
    
    for i = 1:15
    
        clipLimit = clipLimit*2;
        L1 = adapthisteq(L, 'NumTiles', tileSize, 'ClipLimit', clipLimit);     
        LAB1(:,:,1) = L1*100;
        J1 = lab2rgb(LAB1);
        multi = [multi; {J1}];
        
        figure
        subplot(1, 2, 1)
        imshow(J1)
        subplot(1, 2, 2)
        imhist(J1)
        suptitle(['clipLimit = ' num2str(clipLimit)])
    end
    
    figure
    montage(multi);
    title('tile size 4x4, clip limits .000025 - 1')
    
   % Normal histogram equalization
    J = histeq(GRAY_rescaled);
    figure
    montage({GRAY_rescaled J,})
    title('grayscale vs histeq')
    figure
    subplot(1, 2, 1)
    imshow(J)
    subplot(1, 2, 2)
    imhist(J)
    title('histeq')
    
end