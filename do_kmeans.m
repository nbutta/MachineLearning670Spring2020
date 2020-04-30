%%
% File: do_kmeans.m
%   Perform the k-mean computation and plot the segmented image.
% Inputs:
%   - RGB: Image (RGB colorspace)
%   - K: Number of clusters
%
% Outputs:
%   - CA: Cluster Assignments
%   - C: Centroids
%   - SI: Segmented Image
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%
function [CA,C,SI] = do_kmeans(RGB,varargin)
    % Initialize variables
    CA = [];
    C = [];
    SI = [];

    p = inputParser;
    p.KeepUnmatched = true;
    addOptional(p,'K',5); % K=5 default value 
    parse(p,varargin{:});
    
    rows = size(RGB, 1);
    columns = size(RGB, 2);
    num_pixels = rows*columns;
    K = p.Results.K;

    r = double(RGB(:,:,1));
    g = double(RGB(:,:,2));
    b = double(RGB(:,:,3));

    % Reshape the image color channels to rows and concatenate them
    R_row = reshape(r, 1, num_pixels);
    G_row = reshape(g, 1, num_pixels);
    B_row = reshape(b, 1, num_pixels);

    Image = double([R_row;G_row;B_row])';

    % MATLAB kmeans
    [cluster_assignments, centroids] = kmeans(Image, K);

    % Put the segmented image together
    cluster_assignments = cluster_assignments';
    centroids = centroids';

    segmented_image_r = zeros(1, num_pixels);
    segmented_image_g = zeros(1, num_pixels);
    segmented_image_b = zeros(1, num_pixels);

    for j = 1:num_pixels
        segmented_image_r(j) = centroids(1, cluster_assignments(j));
        segmented_image_g(j) = centroids(2, cluster_assignments(j));
        segmented_image_b(j) = centroids(3, cluster_assignments(j));
    end

    sr = reshape(segmented_image_r, rows, columns);
    sg = reshape(segmented_image_g, rows, columns);
    sb = reshape(segmented_image_b, rows, columns);

    segmented_image = zeros(rows, columns, 3);
    segmented_image(:,:,1) = sr;
    segmented_image(:,:,2) = sg;
    segmented_image(:,:,3) = sb;
    
    % Set output variables
    CA = cluster_assignments;
    C = centroids;
    SI = segmented_image;

    % Plot the MATLAB segmented result with K = 5
    % subplot(1, 3, 3)
    % imshow(uint8(segmented_image))
    % title('MATLAB: Segmented Image (K = 5)')
    montage({RGB, uint8(segmented_image)})
end