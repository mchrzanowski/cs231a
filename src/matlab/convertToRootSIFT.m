function descriptors = convertToRootSIFT(descriptors)
    % convert SIFT to RootSIFT.
    % http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
    % 1). L1 normalize.
    % 2). take element-wise sqrt.
    descriptors = descriptors ./ repmat(sum(abs(descriptors), 1), size(descriptors, 1), 1);
    descriptors = sqrt(descriptors);
end