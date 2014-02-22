function fv = generateFisherVector(img_file)

    img = imread(img_file);
    img = rgb2gray(img);
    img = single(img);
    img = imresize(img, [160, 125]);

    [keypts descriptors] = generateDenseSIFTDescriptors(img);
    descriptors = performPCA(descriptors);
    descriptors = [descriptors; keypts];

    [means, cov_diags, priors] = vl_gmm(descriptors, 512);
    fv = vl_fisher(descriptors, means, cov_diags, priors, 'Improved');

end