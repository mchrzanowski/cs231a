function [X U] = performPCA(X)
% perform PCA
% assume X is a 128xD matrix
    X = X - repmat(mean(X, 1), size(X, 1), 1);
    covar = X * X' / size(X, 2);
    [U, ~, ~] = svd(covar, 'econ');
    U = U(:, 1:64);
    X = U' * X;
end
