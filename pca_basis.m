function [eigfaces, eigvals] = pca_basis(A)
%
% This function performs PCA on the passed input data and returns
% the data's PCA vectors and their corresponding eigenvalues.
%
% The i-th eigenvector and i-th eigenvalue should correspond to  each
% other. The eigenvector at index 1 should be the principal (most)
% important eigenvector.

    % Center the data by subtracting the mean of each column (pixel/feature
    % in the matrix. This results in a matrix where each column has 
    % a mean value of zero.
    A = (A-mean(A, 1));

    % Compute the covariance matrix
    C = cov(A);  
    
    % or, manually
    %C = (A')*(A);
    %C = (1/(size(A,1)-1))*C;
       
    % Correlation matrix. Not sure if this is more correct.
    %R = corrcov(C);
   
    % We want to make the covariance matrix diagonal with zeros
    % off-diagonal.
    % Intuitively, that means we would have variance in individual vectors 
    % and no covariance between vectors.
    % The goal is to remove redundancies in the data.
    
    % Lets figure out a new frame of reference
    % The covariance matrix is square symmetric, so we can do an eigen-decomp.
    % that looks like VDV'
 
    % Then, if we compute a new basis, Y = V' * A, then the covariance
    % matrix of Y = YY' = (V'A)(A'V) = V'(cov(A))V = V'(VDV')V = D, which
    % is a diagonal matrix with the eigenvalues of cov(A) on the diagonal.

    % V contains our principal components (eigenfaces).
    % D is diagonal, ordered largest to smallest (eigenvalues/variances)

    % Get the 40 largest magnitude eigenvalues and corresponding eigenvectors
    [V, D] = eigs(double(C), 40, 'lm');
   
    % Normalize the eigenvectors by the corresponding eigenvalues
    eigenvalues = diag(D);
    V = (V./eigenvalues');    

    % Alternatively, we could have used the SVD
    %[Usvd, Ssvd, Vsvd] = svd(C);
    % Y = U' * X with U from SVD => cov(Y) = sigma^2 - diagonal 

    % U allows you to transform to a frame of reference where the cov matix
    % is diagonal with the singular values on the diagonal 

    % U contains principal components (eigenfaces)
    % S is diagonal and contains sigma^2 (eigenvalues)
    % V contains how each image projects onto eigenvectors

    eigfaces = V;
    eigvals = D;

end
