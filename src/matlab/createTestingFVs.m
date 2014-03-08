function [fvs labels] = createTestingFVs(pair_file, data_dir, split, U, M, D, P)
    fvs = [];
    labels = [];

    pairs_per_split = 600;
    data = fileread(pair_file);
    data = regexp(data, '\n', 'split');

    first_pair = (split - 1) * pairs_per_split + 1;
    last_pair = split * pairs_per_split;
    data = data(first_pair:last_pair);

    for i=1:length(data)
        datum = regexp(data{i}, '\t', 'split');
        label = 0;
        if length(datum) == 3       % same person
            first_img = createFVFilename(data_dir, datum{1}, datum{2});
            second_img = createFVFilename(data_dir, datum{1}, datum{3});
            label = 1;
        elseif length(datum) == 4
            first_img = createFVFilename(data_dir, datum{1}, datum{2})
            second_img = createFVFilename(data_dir, datum{3}, datum{4});
            label = -1;
        else
            continue;
        end
        fv1 = generateFisherVector(first_img, U, M, D, P);
        fv2 = generateFisherVector(second_img, U, M, D, P);
        fvs = [fvs fv1 fv2];
        labels = [labels; label size(fvs, 2) - 1, size(fvs, 2)];
    end
end

function fv_filename = createFVFilename(data_dir, person, imgnum)
    imgnum = str2num(imgnum);
    imgnum = sprintf('%.4d', imgnum);
    fv_filename = strcat(data_dir, '/', person, '/', person, '_', imgnum, '.jpg');
end