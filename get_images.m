%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

function A = get_images(basePath, paths, roiX1, roiY1, roiX2, roiY2, num_x_pixels, num_y_pixels)

    num_signs = length(paths);

    A = zeros(num_signs, num_x_pixels*num_y_pixels);

    for i = 1:num_signs

        % Read in the image at the path
        RGB = imread([basePath, char(paths(i))]);
        
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

        RGB_roi = RGB(roiY1(i):roiY2(i), roiX1(i):roiX2(i), :);
        
        % Resize the image to an experimentally-determined size
        RGB_rescaled = imresize(RGB_roi, [num_x_pixels num_y_pixels]);
        
        GRAY = rgb2gray(RGB_rescaled);
        
        A(i, :) = reshape(GRAY, 1, 50*50);
    end
    
end