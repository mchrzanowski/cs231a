function [keypts descriptors] = generateDenseSIFTDescriptors(X)
    % create sift descriptors and keypts
    keypts = [];
    descriptors = [];
    for i=1:5
        [a, b] = vl_dsift(X, 'Size', 8 * sqrt(2) ^ (i - 1), 'Step', 1, 'FloatDescriptors');
        keypts = [keypts a];
        descriptors = [descriptors b];
    end
end
