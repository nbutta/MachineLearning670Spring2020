function [pred, closest_face] = pca_predict(test_face, eigfaces, k, training_faces, training_labels)
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
    % Commented code uses cosine similarity

    closest_label = 1;
    training_scores = training_faces(1,:)*pca_vectors;
    %best_diff = dot(test_scores,training_scores)/(norm(test_scores)*norm(training_scores));
    best_diff = norm(test_scores-training_scores) ^ 2;
    closest_face = training_faces(1,:);

    for i = 2:size(training_faces,1)
        training_scores = training_faces(i,:)*pca_vectors;
        %diff = dot(test_scores,training_scores)/(norm(test_scores)*norm(training_scores));
        diff = norm(test_scores-training_scores) ^ 2;
        %if (diff > best_diff)
        if (diff < best_diff)
            closest_label = i;
            best_diff = diff;
            closest_face = training_faces(i,:);
        end
    end

    pred = training_labels(closest_label);

end