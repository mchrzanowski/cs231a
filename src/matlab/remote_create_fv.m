function remote_create_fv(in1, out1, in2, out2, param_dir)
    if exist('vl_fisher') == 0
        run('~/vlfeat-0.9.18/toolbox/vl_setup.m');
    end
    try
        mean_file = strcat(param_dir, '/means');
        m = dlmread(mean_file);
        cov_file = strcat(param_dir, '/diags');
        d = dlmread(cov_file);
        prior_file = strcat(param_dir, '/priors');
        p = dlmread(prior_file);
        u_file = strcat(param_dir, '/U_matrix');
        u = dlmread(u_file);
        fv1 = generateFisherVector(in1, u, m, d, p);
        fv2 = generateFisherVector(in2, u, m, d, p);
    catch err
        exit(1);
    end
    dlmwrite(out1, fv1);
    dlmwrite(out2, fv2);
    exit(0);
end