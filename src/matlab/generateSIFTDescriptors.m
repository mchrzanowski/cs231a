function [descriptors other_features] = generateSIFTDescriptors(img_file)
    img_height = 160;
    img_width = 125;

    img = imread(img_file);
    img = runViolaJones(img);

    if length(size(img)) == 3
        img = rgb2gray(img);
    end
    img = single(img);
    img = imresize(img, [img_height, img_width]);   % (numrows, numcols)
    skin = imresize(skin, [img_height, img_width]);
    
    [orig_keypts descriptors] = generateDenseSIFTDescriptors(img);
    %descriptors = convertToRootSIFT(descriptors);
    
    % add spatial information.
    keypts = orig_keypts ./ repmat([img_width; img_height], 1, size(descriptors, 2));
    keypts = keypts - 0.5;

    other_features = [keypts; skin_per_keypt];

end
