function [pred, closest_face] = pca_predict(test_face, eigfaces, k, training_scores, training_faces, training_labels)
%
% This function projects the input face and training data onto a 
% k-dimensional subspace spanned by k PCA vectors. The projection
% coordinates are then used to determine the nearest training face
% to the test face as measured by L2 distance in the projected space.
%
% Both the predicted label for the input face and the nearest
% training face are returned.

    pca_vectors = eigfaces(:, 1:k);
    test_scores = test_face*pca_vectors;

    % Find best match using k eigenvectors and Euclidean distance

    scoresz = repmat(test_scores, length(training_labels), 1);
    diffz = scoresz-training_scores;
    normz = vecnorm(diffz') .^ 2;
    [best_diff, i] = min(normz);

    closest_face = training_faces(i,:);
    pred = training_labels(i);
    
end