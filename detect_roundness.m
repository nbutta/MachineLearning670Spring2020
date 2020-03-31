%%
% File: detect_roundness.m
%   Detect roundness in an RGB image and plot the detected results.
% Inputs:
%   - RGB_rescaled: Rescaled Image (RGB colorspace)
%
% Outputs:
%   - None
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%
function output = detect_roundness(RGB_rescaled)

%%  Detect roundness....
    output = [];
    % 
    % % Try boosting contrast first before thresholding
    LAB1 = rgb2lab(RGB_rescaled);
    L = LAB1(:,:,1)/100;
    L1 = adapthisteq(L, 'NumTiles', [4 4], 'ClipLimit', .1);     
    LAB1(:,:,1) = L1*100;
    J1 = lab2rgb(LAB1);
    subplot(3, 5, 2), imshow(J1)

    GRAY = rgb2gray(J1);
    % Identifying round objects for detecting round signs
    %threshold = graythresh(GRAY);
    BW = imbinarize(GRAY);
    notBW = ~BW;

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

    subplot(3, 6, 13), imshow(notBW)
    notBW = bwareaopen(notBW,100);
    subplot(3, 6, 14), imshow(notBW)
    se = strel('disk',2);
    notBW = imclose(notBW,se);
    subplot(3, 6, 15), imshow(notBW)
    notBW = imfill(~notBW,'holes');
    subplot(3, 6, 16), imshow(notBW)
    [notB, notL] = bwboundaries(notBW, 'noholes');
    subplot(3, 6, 17), imshow(notBW)

    subplot(3, 6, 12), imshow(label2rgb(L,@jet,[.5 .5 .5]))
    hold on
    for k = 1:length(B)
      boundary = B{k};
      plot(boundary(:,2),boundary(:,1),'w','LineWidth',2)
    end

    stats = regionprops(L,'Area','Centroid');

    threshold = 0.84;

    % loop over the boundaries
    for k = 1:length(B)

      % obtain (X,Y) boundary coordinates corresponding to label 'k'
      boundary = B{k};

      % compute a simple estimate of the object's perimeter
      delta_sq = diff(boundary).^2;    
      perimeter = sum(sqrt(sum(delta_sq,2)));

      % obtain the area calculation corresponding to label 'k'
      area = stats(k).Area;

      % compute the roundness metric
      metric = 4*pi*area/perimeter^2;

      % display the results
      metric_string = sprintf('%2.2f',metric);

      % mark objects above the threshold with a black circle
      if metric > threshold
        centroid = stats(k).Centroid;
        plot(centroid(1),centroid(2),'ko');
        title('Sign is round!')
      else 
        title('Sign is NOT round!')
      end

      text(boundary(1,2)-10,boundary(1,1)+10,metric_string,...
           'FontSize',8,'FontWeight','bold')

    end

    subplot(3, 6, 18), imshow(label2rgb(notL,@jet,[.5 .5 .5]))
    hold on
    for k = 1:length(notB)
      boundary = notB{k};
      plot(boundary(:,2),boundary(:,1),'w','LineWidth',2)
    end

    stats = regionprops(notL,'Area','Centroid');

    % loop over the boundaries
    for k = 1:length(notB)

      % obtain (X,Y) boundary coordinates corresponding to label 'k'
      boundary = notB{k};

      % compute a simple estimate of the object's perimeter
      delta_sq = diff(boundary).^2;    
      perimeter = sum(sqrt(sum(delta_sq,2)));

      % obtain the area calculation corresponding to label 'k'
      area = stats(k).Area;

      % compute the roundness metric
      metric = 4*pi*area/perimeter^2;

      % display the results
      metric_string = sprintf('%2.2f',metric);

      % mark objects above the threshold with a black circle
      if metric > threshold
        centroid = stats(k).Centroid;
        plot(centroid(1),centroid(2),'ko');
        title('Sign is round!')
      else 
        title('Sign is NOT round!')
      end

      text(boundary(1,2)-10,boundary(1,1)+10,metric_string,...
           'FontSize',8,'FontWeight','bold')

    end


    title(['Metrics closer to 1 indicate that ',...
           'the object is approximately round'])


%STATS = regionprops(L, 'all'); % we need 'BoundingBox' and 'Extent'
% figure,
% imshow(RGB),
% title('Results');
% hold on
% for i = 1 : length(STATS)
%   W(i) = uint8(abs(STATS(i).BoundingBox(3)-STATS(i).BoundingBox(4)) < 0.1);
%   W(i) = W(i) + 2 * uint8((STATS(i).Extent - 1) == 0 );
%   centroid = STATS(i).Centroid;
%   switch W(i)
%       case 1
%           plot(centroid(1),centroid(2),'wO');
%       case 2
%           plot(centroid(1),centroid(2),'wX');
%       case 3
%           plot(centroid(1),centroid(2),'wS');
%   end
% end

%fgetl(fID); % discard line with column headers
    
%f = textscan(fID, '%*d %*d %d %d %d %d %d', 'Delimiter', ';');

end