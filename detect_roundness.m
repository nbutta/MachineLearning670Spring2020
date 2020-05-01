%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% Function to compute roundness of the signs...
% Needs some work still

function features = detect_roundness(basePath, paths, roiX1, roiY1, roiX2, roiY2)

    num_signs = length(paths);

    features = zeros(num_signs, 1);

    for i = 1:num_signs
        
        % Read in the image at the path
        RGB = imread([basePath, char(paths(i))]);

        %RGB_roi = RGB(roiY1(i):roiY2(i), roiX1(i):roiX2(i), :);
        
        % Resize the image to an experimentally-determined size
        %RGB_rescaled = imresize(RGB_roi, [50 50]);
        
        GRAY = rgb2gray(RGB);
        % Identifying round objects for detecting round signs
        %threshold = graythresh(GRAY);
        BW = imbinarize(GRAY);

        figure
        
        subplot(3, 6, 1)
        imshow(RGB)
        
        subplot(3, 6, 2)
        imshow(GRAY)
        
        subplot(3, 6, 7)
        imshow(BW)
        
        BW = bwareaopen(BW,100);
        subplot(3, 6, 8)
        imshow(BW)
        
        se = strel('disk',2);
        BW = imclose(BW,se);
        subplot(3, 6, 9)
        imshow(BW)
        
        BW = imfill(BW,'holes');
        subplot(3, 6, 10)
        imshow(BW)
        
        [B,L] = bwboundaries(BW, 'noholes');
        subplot(3, 6, 11)
        imshow(BW)

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
          %uiwait(msgbox(message));
        end
    end

end
