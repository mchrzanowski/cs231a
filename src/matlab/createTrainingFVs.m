function [fvs labels] = createTrainingFVs(data_dir, U, M, D, P)
    fvs = [];
    labels = [];

    current_person = 0;
    people = dir(strcat(data_dir, '*'));
    assert(~isempty(people));
    h = waitbar(0,'Loading People...');
    length(people)
    for person = people'
        if person.name(1) == '.'
            continue;
        end
        current_person = current_person + 1;
        f_q_person = strcat(data_dir, person.name, '/');
        images = dir(strcat(f_q_person, '*.jpg'));
        waitbar((current_person/length(people)),h);
        
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_person, image.name);
<<<<<<< HEAD
            fv = generateFisherVector(f_q_img_path, U, M, D, P);
=======
            [descriptors, keypts] = generateSIFTDescriptors(f_q_img_path);
            data = U' * descriptors;
            data = [data; keypts];
            data = double(data);
            fv = vl_fisher(data, M, D, P, 'Improved');
>>>>>>> 8986e8531a89306940cb27e37cfdf12e847a76cb
            fvs = [fvs fv];
            labels = [labels current_person];
        end
    end
end