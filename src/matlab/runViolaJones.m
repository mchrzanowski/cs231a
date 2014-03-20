function img = runViolaJones(img)
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
end