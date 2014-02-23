function X = performPCA(X)
% perform PCA
% assume X is a 128xD matrix
% create a whitened 64xD matrix
    X = X - repmat(mean(X, 1), size(X, 1), 1);
    covar = X * X' / size(X, 2);
    [U, ~, ~] = svd(covar, 'econ');
    X = U(:, 1:64)' * X;
end
