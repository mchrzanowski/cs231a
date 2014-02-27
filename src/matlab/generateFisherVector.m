function fv = generateFisherVector(img_file)
    img_height = 160;
    img_width = 125;

    img = imread(img_file);
    img = rgb2gray(img);
    img = single(img);
    img = imresize(img, [img_height, img_width]);   % (numrows, numcols)

    %[keypts descriptors] = generateDenseSIFTDescriptors(img);
    try
        s = 7.5;
        [keypts descriptors] = vl_phow(img, 'Sizes', [s, s * sqrt(2), s * 2, s * sqrt(2) ^ 3, s * sqrt(2) ^ 4], 'Step', 1, 'FloatDescriptors', true);

        descriptors = convertToRootSIFT(descriptors);
        
        % add spatial information.
        keypts = keypts(1:2, :);    % x & y coords.
        keypts = keypts ./ repmat([img_width; img_height], 1, size(descriptors, 2));
        keypts = keypts - 0.5;

        descriptors = performPCA(descriptors);
        descriptors = [descriptors; keypts];

        [means, cov_diags, priors] = vl_gmm(descriptors, 512);
        fv = vl_fisher(descriptors, means, cov_diags, priors, 'Improved');
    catch err
        fv = 0;
    end
end