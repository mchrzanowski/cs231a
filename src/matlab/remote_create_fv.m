function remote_create_fv(in, out)
    fv = generateFisherVector(in);
    dlmwrite(out, fv);
    exit();
end