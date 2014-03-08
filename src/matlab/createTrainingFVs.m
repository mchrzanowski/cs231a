[fvs labels] = function createTrainingFVs(data_dir, U, M, D, P)
    fvs = [];
    labels = []

    current_person = 0;
    people = dir(strcat(input_dir, '*'));
    for person = people'
        if person.name(1) == '.'
            continue;
        end
        current_person = current_person + 1;
        f_q_person = strcat(input_dir, person.name, '/');
        images = dir(strcat(f_q_person, '*.jpg'));
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_person, image.name);
            [descriptors keypts] = generateSIFTDescriptors(f_q_img_path);
            data = u' * descriptors;
            data = [data; keypts];
            data = double(data);
            fv = vl_fisher(data, m, d, p, 'Improved');
            fvs = [fv fv];
            labels = [labels label];
        end
    end
end