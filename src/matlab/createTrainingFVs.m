function [fvs labels] = createTrainingFVs(data_dir, U, M, D, P)
    fvs = [];
    labels = [];

    current_person = 0;
    people = dir(strcat(data_dir, '*'));
    assert(~isempty(people));
    %h = waitbar(0,'Loading People...');
    length(people)
    for person = people'
        if person.name(1) == '.'
            continue;
        end
        current_person = current_person + 1;
        f_q_person = strcat(data_dir, person.name, '/');
        images = dir(strcat(f_q_person, '*.jpg'));
        %waitbar((current_person/length(people)),h);
        if(mod(current_person,200)==0)
           (current_person/length(people))
        end
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_person, image.name);
            fv = generateFisherVector(f_q_img_path, U, M, D, P);
            fvs = [fvs fv];
            labels = [labels current_person];
        end
    end
end
