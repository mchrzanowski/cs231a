%Train and test the neural network procedure
clear all; close all; clc;

num_iters=10^8;

image_dir = '/Users/francoischaubard/Desktop/lfw-deepfunneled/';
U=dlmread('../../params/gmm_params_rooted_df/U_matrix');
M=dlmread('../../params/gmm_params_rooted_df/means');
D=dlmread('../../params/gmm_params_rooted_df/diags');
P=dlmread('../../params/gmm_params_rooted_df/priors');

% generate all the fv's
[fvs labels] = createTrainingFVs(image_dir, U, M, D, P);

unique_people
people_with_more_than_one
for i=1:num_iters
    
    % get the two fvs
    if (rand()>0.5)%pull same person fvs
        
    else % pull 2 different people fvs
        
    end
    
    
    
end