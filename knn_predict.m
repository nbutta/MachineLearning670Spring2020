%  KNN function to predict a digit using the k-nearest neighbor
%  Inputs:
%       k - Number of neighbors
%       TRAIN_IMAGES - Training data set
%       TRAIN_LABELS - Training data labels
%       TEST_IMAGES - Test data set
%  Outputs:
%       PRED_LABELS - Predicted Labels
%  Author Name: Johanna Rivera
function [PRED_LABELS] = knn_predict(k, TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES)

    PRED_LABELS = [];
    numTestImages = size(TEST_IMAGES,1);
    numTrainImages = length(TRAIN_LABELS);
    
    % euclidian distances
    d = zeros(numTestImages,numTrainImages);
    % indeces
    indcs = zeros(numTestImages,numTrainImages);
    
     % Compute the euclidian distances
    for testIdx=1:numTestImages
        for trainIdx=1:numTrainImages
            % Compute the euclidian distance between the two images
            d(testIdx,trainIdx)= norm(...
                TEST_IMAGES(testIdx,:)-TRAIN_IMAGES(trainIdx,:),1);
        end
        % sort distances from low to high
        [d(testIdx,:),indcs(testIdx,:)]=sort(d(testIdx,:),'ascend');
    end
    
    % Get the nearest neighbors
    k_neighbors = indcs(:,1:k);
    
    % Get the majority vote
    for ii=1:numTestImages
        PRED_LABELS(ii)=mode(TRAIN_LABELS(k_neighbors(ii,:)));
    end
end