function recon = pca_reconstruction(face, eigfaces, k)
%
% This function projects the input face onto a k-dimensional subspace
% spanned by k PCA vectors. The projection coordinates are then used
% to reconstruct the input face.
%
% "recon" is a flattened vector of the same length as "face".

% Get the k = 10, 20, 30, 40 eigenvectors
% size is 10304 x k
pca_vectors = eigfaces(:, 1:k);

% Project the chosen face onto the k = 10, 20, 30, 40 eigenvectors
% Basically, dot product of the face with each eigenface
% These are the value of the weights used for reconstruction of the face
% from the weighted eigenfaces
% size is 1 x k
proj = face*pca_vectors;

% Reconstruct the face from the projections
% size is 10304 x 1
recon = pca_vectors*proj';

% size is 1 x 10304, same as face
recon = recon';

end