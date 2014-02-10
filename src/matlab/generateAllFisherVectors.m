function generateAllFisherVectors
    input_dir = '/opt/cs231a/lfw/';
    output_dir = '/opt/cs231a/serialized/fvs/';
    mkdir(output_dir);

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
            f_q_output_path = strcat(output_dir, image.name);
            if exist(f_q_output_path, 'file') ~= 0
                continue;
            end
            fv = generateFisherVector(f_q_img_path);
            fv = fv / norm(fv, 2);
            dlmwrite(f_q_output_path, fv);
        end
    end
end
