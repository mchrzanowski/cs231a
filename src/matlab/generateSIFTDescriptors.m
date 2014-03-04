function [descriptors keypts] = generateSIFTDescriptors(img_file)
    img_height = 160;
    img_width = 125;

    img = imread(img_file);
    img = rgb2gray(img);
    img = single(img);
    img = imresize(img, [img_height, img_width]);   % (numrows, numcols)

    [keypts descriptors] = generateDenseSIFTDescriptors(img);
    descriptors = convertToRootSIFT(descriptors);

    % add spatial information.
    keypts = keypts(1:2, :);    % x & y coords.
    keypts = keypts ./ repmat([img_width; img_height], 1, size(descriptors, 2));
    keypts = keypts - 0.5;

end