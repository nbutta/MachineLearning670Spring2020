%%
% File: svm_predict.m
%   Classify the test data set using the svm model
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%   Term: Spring 2020
%   Author: Johanna Rivera
%
function [p_labels] = svm_predict(mdl, data)

    % predict the labels
    p_labels = predict(mdl,data);
    
end