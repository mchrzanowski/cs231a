function getAllSIFTDescriptors(input_dir, output_dir)
    mkdir(output_dir);

    X = [];

    subfolders = dir(strcat(input_dir, '*'));
    for subfolder = subfolders'
        if subfolder.name(1) == '.'
            continue;
        end
        disp(sprintf('Generating fvs for: %s', subfolder.name));
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
            %if exist(f_q_output_path_desc, 'file') ~= 0
            %    continue;
            %end
            [descriptors keypts] = generateSIFTDescriptors(f_q_img_path);
            dlmwrite(f_q_output_path_desc, descriptors);
            dlmwrite(f_q_output_path_keypt, keypts);
            X = [X descriptors];
        end
    end

    U_output = strcat(output_dir, '/U_matrix');
    [X U] = performPCA(X);
    dlmwrite(U_output, U);
    [means, cov_diags, priors] = vl_gmm(X, 512);

    mean_output = strcat(output_dir, '/means');
    dlmwrite(mean_output, means);

    diag_output = strcat(output_dir, '/diags');
    dlmwrite(diag_output, cov_diags);

    prior_output = strcat(output_dir, '/priors');
    dlmwrite(prior_output, priors);

end