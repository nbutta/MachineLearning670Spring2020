%%
% File: svm_trai.m
%   Train the SVM classifier using the linear kernel function
%
% Info:
%   Class: EN.525.670.81 - Machine Learning for Signal Processing
%   Term: Spring 2020
%   Author: Johanna Rivera
%
function [mdl] = svm_train(data,labels)
    
    % use built-in matlab
    % create model template
    t_svm = templateSVM('Standardize', true,...
    'KernelFunction','linear');

    % train the model
    mdl = fitcecoc(data,labels,'Learners',t_svm);
    
end