function [descriptors keypts] = generateSIFTDescriptors(img_file)
    img_height = 160;
    img_width = 125;

    img = imread(img_file);

    % viola jones detector.
    faceDetector = vision.CascadeObjectDetector;
    bboxes = step(faceDetector, img);
    if ~isempty(bboxes)
        min_x = min(bboxes(:, 1));
        min_y = min(bboxes(:, 2));
        max_x = max(bboxes(:, 1) + bboxes(:, 3));
        max_y = max(bboxes(:, 2) + bboxes(:, 4));

        min_x = max(1, min_x);
        min_y = max(1, min_y);
        max_x = min(size(img, 2), max_x);
        max_y = min(size(img, 1), max_y);
        if max_x > min_x && max_y > min_y
            img = img(min_y:max_y, min_x:max_x, :);
        end
    end

    if length(size(img)) == 3
        img = rgb2gray(img);
    end
    img = single(img);
    img = imresize(img, [img_height, img_width]);   % (numrows, numcols)

    img_height = min(img_height, size(img, 1));
    img_width = min(img_width, size(img, 2));
    
    [keypts descriptors] = generateDenseSIFTDescriptors(img);
    %descriptors = convertToRootSIFT(descriptors);

    % add spatial information.
    keypts = keypts(1:2, :);    % x & y coords.
    keypts = keypts ./ repmat([img_width; img_height], 1, size(descriptors, 2));
    keypts = keypts - 0.5;

end