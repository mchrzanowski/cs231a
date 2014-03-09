function remote_create_fv(in, out, param_dir)
    try
        mean_file = strcat(param_dir, '/means');
        m = dlmread(mean_file);
        cov_file = strcat(param_dir, '/diags');
        d = dlmread(cov_file);
        prior_file = strcat(param_dir, '/priors');
        p = dlmread(prior_file);
        u_file = strcat(param_dir, '/U_matrix');
        u = dlmread(u_file);
        fv = generateFisherVector(in, u, m, d, p);
    catch err
        fv = 2;
        exit(1);
    end
    dlmwrite(out, fv);
    exit(0);
end