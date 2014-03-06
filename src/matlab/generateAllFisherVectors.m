function generateAllFisherVectors(input_dir, param_dir, output_dir)
% i assume you've already run getAllSIFTDescriptors !
    
    mkdir(output_dir);

    mean_file = strcat(param_dir, '/means');
    m = dlmread(mean_file);

    cov_file = strcat(param_dir, '/diags');
    d = dlmread(cov_file);

    prior_file = strcat(param_dir, '/priors');
    p = dlmread(prior_file);

    u_file = strcat(param_dir, '/U_matrix');
    u = dlmread(u_file);

    subfolders = dir(strcat(input_dir, '*'));
    for subfolder = subfolders'
        if subfolder.name(1) == '.'
            continue;
        end
        disp(sprintf('Generating FV for: %s', subfolder.name));
        f_q_subfolder = strcat(input_dir, subfolder.name, '/');
        images = dir(strcat(f_q_subfolder, '*.jpg'));
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_subfolder, image.name);
            
            f_q_output_path_fv = strcat(output_dir, image.name);
            f_q_output_path_fv = strrep(f_q_output_path_fv, '.jpg', '_fv.csv');
            if exist(f_q_output_path_fv, 'file') ~= 0
                continue;
            end
            [descriptors keypts] = generateSIFTDescriptors(f_q_img_path);
            data = u' * descriptors;
            data = [data; keypts];
            data = double(data);
            fv = vl_fisher(data, m, d, p, 'Improved');
            dlmwrite(f_q_output_path_fv, fv);
        end
    end
end
