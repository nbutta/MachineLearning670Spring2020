%% Nicholas Butta
%  525.670 Machine Learning for Signal Processing
%  Spring 2020

function predicted_classes = knn(K, train_scores, trainClasses, test_scores)

    % Do KNN
    dims = size(test_scores);
    num_test_images = dims(1);
    num_train_images = length(trainClasses);

    % Find best match using k eigenvectors and Euclidean distance
    closest_sign = zeros(num_test_images, 1);

    % Compute distance from each test image to all other training images
    for test_idx = 1:num_test_images
        temp = repmat(test_scores(test_idx,:), num_train_images, 1);
        diffz = temp-train_scores;

        % Find the 2 norm of each of the rows.
        normz = vecnorm(diffz, 2, 2);

        % Get the smallest K values of the norms
        [B, I] = mink(normz, K);

        % Get the classes of the K closest neighbors
        labels = trainClasses(I);

        % The majority of the labels is the closest digit
        % Tie goes to the smallest value
        closest_sign(test_idx) = mode(labels);

    end
    
    predicted_classes = closest_sign;

end