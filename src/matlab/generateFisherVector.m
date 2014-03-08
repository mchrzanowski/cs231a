function fv = generateFisherVector(filename, U, M, D, P)
    [descriptors keypts] = generateSIFTDescriptors(filename);
    data = U' * descriptors;
    data = [data; keypts];
    data = double(data);
    fv = vl_fisher(data, M, D, P, 'Improved');
end