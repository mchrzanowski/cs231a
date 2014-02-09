function X = pcaAndWhiten(X)
% whiten & perform PCA
% assume X is a 128xD matrix
% create a whitened 64xD matrix
    epsilon = 1e-4;
    X = X - repmat(mean(X, 1), size(X, 1), 1);
    covar = X * X' / size(X, 2);
    [u, s, ~] = svd(covar, 'econ');
    s = s(1:64, 1:64);
    u = u(:, 1:64);
    X = diag(1 ./ sqrt(diag(s) + epsilon)) * u' * X;
end