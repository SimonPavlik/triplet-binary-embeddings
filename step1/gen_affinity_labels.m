function triplet_samples_idxes = gen_affinity_labels(train_data)


disp('gen_affinity_labels...');

e_num=size(train_data.label_data, 1);
label_data=train_data.label_data;
assert(size(label_data, 1)==e_num);
assert(size(label_data, 2)==1);

maximum_sample_num = 5;

triplet_samples_idxes = ones(maximum_sample_num * e_num, 3);

for e_idx = 1:e_num

    relevant_sel=label_data(e_idx)==label_data;
    irrelevant_sel=~relevant_sel;
    relevant_sel(e_idx)=false;

    relevant_idxes=find(relevant_sel);
    irrelevant_idxes=find(irrelevant_sel);

    sub_relevant_idxes = randsample(relevant_idxes, maximum_sample_num);
    sub_irrelevant_idxes = randsample(irrelevant_idxes, maximum_sample_num);

    bias = e_idx * ones(maximum_sample_num, 1);


    sub_triplet_samples_idxes = [bias, sub_relevant_idxes, sub_irrelevant_idxes];

    if e_idx == 1
        triplet_samples_idxes(1:maximum_sample_num, :) = sub_triplet_samples_idxes;
    else
        triplet_samples_idxes((e_idx - 1) * maximum_sample_num + 1:e_idx * maximum_sample_num, :) = sub_triplet_samples_idxes;

    end
end

end
