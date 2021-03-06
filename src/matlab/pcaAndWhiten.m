function X = pcaAndWhiten(X)
% whiten & perform PCA
% assume X is a 128xD matrix
% create a whitened 64xD matrix
    epsilon = 1e-10;
    X = X - repmat(mean(X, 1), size(X, 1), 1);
    covar = X * X' / size(X, 2);
    [U, S, ~] = svd(covar, 'econ');
    S = S(1:64, 1:64);
    U = U(:, 1:64);
    X = diag(1 ./ sqrt(diag(S) + epsilon)) * U' * X;
end