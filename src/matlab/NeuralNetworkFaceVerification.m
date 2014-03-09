%Train and test the neural network procedure
clear all; close all; clc;

curr_dir = pwd;
cd('~/vlfeat-0.9.18/toolbox/')
vl_setup
cd(curr_dir)

num_iters=10^6;

image_dir = '~/lfw-deepfunneled/';
U=dlmread('../../params/gmm_params_rooted_df/U_matrix');
M=dlmread('../../params/gmm_params_rooted_df/means');
D=dlmread('../../params/gmm_params_rooted_df/diags');
P=dlmread('../../params/gmm_params_rooted_df/priors');

display('checking for fvs')
fv_file = sprintf('%s/fvs_francois.mat',pwd());

if exist(fv_file, 'file') ~= 2
	% generate all the fv's
	display('fvs didnt exist! Creating them..')
	[fvs labels] = createTrainingFVs(image_dir, U, M, D, P);
	save('fvs_francois.mat','fvs');
	save('labels_francois.mat','labels');
else
	display('fvs did exist! Loading them..')
	load('fvs_francois.mat');
	load('labels_francois.mat');

end

unique_people = unique(labels);
histogram = histc(labels,1:labels(end));
moreThan1Picture = find(histogram>1);

fv_diff=[];

for i=1:num_iters
    
    % get the two fvs
    if (rand()>0.5)%pull same person fvs
        person_idx = datasample(moreThan1Picture,1,'Replace',false);
        possible_choices = find(labels==person_idx);
        choices = datasample(possible_choices,2,'Replace',false);
        fv1 = fvs(:,choices(1));
        fv2 = fvs(:,choices(2));
        
    else % pull 2 different people fvs
        person_idxs = datasample(unique_people,2,'Replace',false);
        choices = [datasample(find(labels==person_idxs(1)),1,'Replace',false), ...
                   datasample(find(labels==person_idxs(2)),1,'Replace',false)];
        fv1 = fvs(:,choices(1));
        fv2 = fvs(:,choices(2));
    end
    fv_diff=[fv_diff sparse(fv1-fv2)];
    
    
    
    
end



