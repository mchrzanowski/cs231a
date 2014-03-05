function generateAllFisherVectors(input_dir, output_dir)
% i assume you've already run getAllSIFTDescriptors !

    mean_file = strcat(input_dir, '/means');
    m = dlmread(mean_file);

    cov_file = strcat(input_dir, '/diags');
    d = dlmread(cov_file);

    prior_file = strcat(input_dir, '/priors');
    p = dlmread(prior_file);

    u_file = strcat(output_dir, '/U_matrix');
    u = dlmread(u_file);

    imgs = dir(strcat(input_dir, '*_descriptor.csv'));
    for img = imgs'
        if img.name(1) == '.'
            continue;
        end

        fv_output_file = strcat(output_dir, strrep(img, '_descriptor', '_fv'));
        if exist(fv_output_file, 'file') ~= 0
            continue;
        end
        
        disp(sprintf('Generating fv for: %s', img.name));
        descriptors = dlmread(img);
        keypt_file = strcat(input_dir, strrep(img, '_descriptor', '_keypts'));
        keypts = dlmread(keypt_file);

        data = u' * descriptors;
        data = [data; keypts];

        fv = vl_fisher(data, m, d, p, 'Improved');
        dlmwrite(fv_output_file, fv);
    end
end
