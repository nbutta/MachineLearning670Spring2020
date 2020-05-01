%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

% For each of the grayscale images, perform histogram equalization
% and return a matrix of the newly equalized images.

function output = boost_gray_contrast(images)

    dims = size(images);
    num_images = dims(1);
    num_pixels = dims(2);
    
    output = zeros(dims);
    
    % Assume the images are square
    x_pixels = sqrt(num_pixels);
    y_pixels = sqrt(num_pixels);
    
    for i = 1:num_images
       
        img = images(i,:);        
        img = reshape(img, x_pixels, y_pixels);
        
% %         figure
% %         imshow(img)
        
        img = histeq(uint8(img));
        
%         figure
%         imshow(img)
        
        output(i,:) = reshape(img, 1, num_pixels);
    end

%     tileSize = [4 4];
%     clipLimit = .000025;
%     multi = {RGB_rescaled};
%     LAB1 = rgb2lab(RGB_rescaled);
%     L = LAB1(:,:,1)/100;
%     
%     for i = 1:15
%     
%         clipLimit = clipLimit*2;
%         L1 = adapthisteq(L, 'NumTiles', tileSize, 'ClipLimit', clipLimit);     
%         LAB1(:,:,1) = L1*100;
%         J1 = lab2rgb(LAB1);
%         multi = [multi; {J1}];
%         
%         figure
%         subplot(1, 2, 1)
%         imshow(J1)
%         subplot(1, 2, 2)
%         imhist(J1)
%         suptitle(['clipLimit = ' num2str(clipLimit)])
%     end
%     
%     figure
%     montage(multi);
%     title('tile size 4x4, clip limits .000025 - 1')

end