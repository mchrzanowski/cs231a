function [keypts descriptors] = generateDenseSIFTDescriptors(X)
    % create sift descriptors and keypts
    keypts = [];
    descriptors = [];
    sizes = [4; 13; 18; 26; 37];
    for i=1:length(sizes)
        [a, b] = vl_dsift(X, 'Size', sizes(i), 'Step', 1, 'FloatDescriptors');
        keypts = [keypts a];
        descriptors = [descriptors b];
    end
end