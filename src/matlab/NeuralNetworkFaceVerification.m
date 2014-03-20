%Train and test the neural network procedure
clear all; close all; clc;
matlabpool close force local;
matlabpool open 12;
c = clock;
disp(datestr(datenum(c(1),c(2),c(3),c(4),c(5),c(6))));
curr_dir = pwd;
cd('~/vlfeat-0.9.18/toolbox/')
vl_setup
cd(curr_dir)

num_iters=10^6;
params_dir ='~/non_root_df_41k'
image_dir = '~/lfw-deepfunneled/';

display('checking for fvs')
fv_file = sprintf('%s/fvs_francois.mat',pwd());
fv_diff_file = sprintf('%s/fv_diff.mat',pwd());

if exist(fv_diff_file, 'file') == 2
        fv_diff=load('fv_diff.mat');
        training_labels=load('training_labels.mat');

        % generate all the fv's
        display('fv_diff did exist! Loading them..')
else

 if exist(fv_file, 'file') ~= 2

        U=dlmread(sprintf('%s/U_matrix',params_dir));
	M=dlmread(sprintf('%s/means',params_dir));
	D=dlmread(sprintf('%s/diags',params_dir));
	P=dlmread(sprintf('%s/priors',params_dir));


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

 training_labels=[];
 display('starting to build fv_diff dataset')
 parfor i=1:num_iters
    
    % get the two fvs
    if (rand()>0.5)%pull same person fvs
        person_idx = datasample(moreThan1Picture,1,'Replace',false);
        possible_choices = find(labels==person_idx);
        choices = datasample(possible_choices,2,'Replace',false);
        fv1 = fvs(:,choices(1));
        fv2 = fvs(:,choices(2));
        same=1;
    else % pull 2 different people fvs
        person_idxs = datasample(unique_people,2,'Replace',false);
        choices = [datasample(find(labels==person_idxs(1)),1,'Replace',false), ...
                   datasample(find(labels==person_idxs(2)),1,'Replace',false)];
        fv1 = fvs(:,choices(1));
        fv2 = fvs(:,choices(2));
        same=-1;
    end
    fv_diff=[fv_diff sparse(fv1-fv2)]; 
    training_labels=[training_labels same];
    
 end
 save('/tmp/fv_diff6.mat','fv_diff');
 save('/tmp/training_labels6.mat','training_labels');
end
c=clock;
disp(datestr(datenum(c(1),c(2),c(3),c(4),c(5),c(6))));
display('training svm..')
%training_labels = sparse(training_labels);
%model_params = '-c 1 -g 0.07';
addpath('~/liblinear-1.94/matlab/');
model_params =  '-c  1';
%model_params = '-t 0';

split_amount = floor(0.75*size(training_labels,2));
addpath('~/libsvm-3.17/matlab');
model = train(training_labels(1:split_amount)', sparse(fv_diff(:,1:split_amount)'), model_params);
disp(datestr(datenum(c(1),c(2),c(3),c(4),c(5),c(6))));
display('testing svm..')
test_fvs=sparse(fv_diff(:,(split_amount+1):end));
testing_labels=training_labels((split_amount+1):end);
%[test_fvs testing_labels] = createTestingFVs(pair_file, data_dir, split, U, M, D, P)

[predicted_label, accuracy, decision_values] = predict(testing_labels',sparse( test_fvs'), model);
disp(datestr(datenum(c(1),c(2),c(3),c(4),c(5),c(6))));
display('finished testing svm..')
accuracy
tp_plus_tn = (predicted_label==testing_labels');
tp = sum(predicted_label(tp_plus_tn)==1);
tn = sum(predicted_label(tp_plus_tn)==-1);
total = length(predicted_label);
fp=total-tp;
fn=total-tn;
p=tp/(fp+tp)
r=tp/(tp+fn)
tp
tn
fp
fn
tp_plus_tn
total
sum(predicted_label==testing_labels')/length(testing_labels)

%cd(curr_dir)
