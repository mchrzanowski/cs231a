function generateGMMParams(input_dir, output_dir)
    mkdir(output_dir);

    X = [];
    Y = [];

    subfolders = dir(strcat(input_dir, '*'));
    for subfolder = subfolders'
        if subfolder.name(1) == '.'
            continue;
        end
        disp(sprintf('Generating SIFT descriptors for: %s', subfolder.name));
        f_q_subfolder = strcat(input_dir, subfolder.name, '/');
        images = dir(strcat(f_q_subfolder, '*.jpg'));
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_subfolder, image.name);
            f_q_output_path_desc = strcat(output_dir, image.name);
            f_q_output_path_keypt = strcat(output_dir, image.name);
            
            f_q_output_path_desc = strrep(f_q_output_path_desc, '.jpg', '_descriptor.csv');
            f_q_output_path_keypt = strrep(f_q_output_path_keypt, '.jpg', '_keypts.csv');
            
            [descriptors other_features] = generateSIFTDescriptors(f_q_img_path);

            indx = randsample(1:length(descriptors), 95);

            X = [X descriptors(:, indx)];
            Y = [Y other_features(:, indx)];
        end
    end

    U_output = strcat(output_dir, '/U_matrix');
    [X U] = performPCA(X);
    dlmwrite(U_output, U);

    X = [X; Y];
    [means, cov_diags, priors] = vl_gmm(X, 512);

    mean_output = strcat(output_dir, '/means');
    dlmwrite(mean_output, means);

    diag_output = strcat(output_dir, '/diags');
    dlmwrite(diag_output, cov_diags);

    prior_output = strcat(output_dir, '/priors');
    dlmwrite(prior_output, priors);

end