function [descriptors other_features] = generateSIFTDescriptors(img_file)
    img_height = 160;
    img_width = 125;

    img = imread(img_file);

    img = runViolaJones(img);
    %skin = generate_skinmap(img);

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

    % scrub bad data.
    %skin_per_keypt = assignSkinToDescriptors(skin, orig_keypts);
    bad_indices = scrubDescriptors(skin_per_keypt);
    descriptors(:, bad_indices) = [];
    keypts(:, bad_indices) = [];
    %skin_per_keypt(:, bad_indices) = [];

    %other_features = [keypts; skin_per_keypt];

    %assert(sum(isnan(other_features(:))) == 0 && sum(isinf(other_features(:))) == 0 && ...
    %    sum(isnan(descriptors(:))) == 0 && sum(isinf(descriptors(:))) == 0);

end

function skin_per_keypt = assignSkinToDescriptors(skin, orig_keypts)
    total_skin = 1e-12 + sum(skin(:));
    x_length = round(orig_keypts(1, 1) - 1);
    y_length = round(orig_keypts(2, 1) - 1);
    skin_per_keypt = zeros(1, size(orig_keypts, 2));
    for i=1:size(orig_keypts, 2)
        min_x = max(1, round(orig_keypts(1, i) - x_length));
        min_y = max(1, round(orig_keypts(2, i) - y_length));

        max_x = min(round(orig_keypts(1, i) + x_length), size(skin, 2));
        max_y = min(round(orig_keypts(2, i) + y_length), size(skin, 1));
        skin_per_keypt(i) = sum(sum(skin(min_y:max_y, min_x:max_x))) / total_skin;
    end
end

function bad_indices = scrubDescriptors(descriptors)
    nans = all(isnan(descriptors), 1);
    infs = all(isinf(descriptors), 1);
    bad_indices = or(nans, infs);
end
