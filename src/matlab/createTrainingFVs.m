function [fvs labels] = createTrainingFVs(data_dir, U, M, D, P)
    fvs = [];
    labels = [];

    %current_person = 0;
    people = dir(strcat(data_dir, '*'));
    assert(~isempty(people));
    %h = waitbar(0,'Loading People...');
    nn=length(people)
    parfor current_person=1:length(people)
        person=people(current_person);
        if person.name(1) == '.'
            continue;
        end
        %current_person = current_person + 1;
        f_q_person = strcat(data_dir, person.name, '/');
        images = dir(strcat(f_q_person, '*.jpg'));
        %waitbar((current_person/length(people)),h);
        if(mod(current_person,20)==0)
           display((current_person))
        end
        for image = images'
            if image.name(1) == '.'
                continue;
            end
            f_q_img_path = strcat(f_q_person, image.name);
            fv = generateFisherVector(f_q_img_path, U, M, D, P);
            fv_sparse = sparse(fv)
            fvs = [fvs, fv_sparse];
            labels = [labels current_person];
        end
    end
end
