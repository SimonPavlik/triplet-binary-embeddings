train_save_path = '。。。。。/data/cifar_10/train';
test_save_path = '。。。。。/data/cifar_10/test';

train_img_path = '。。。。。/data/cifar_10/images/train';
test_img_path = '。。。。。/data/cifar_10/images/test/test_batch.mat';

%% process train data
last_img = 0;
for batch = 1:5
    load(strcat(train_img_path,'/data_batch_', num2str(batch), '.mat'));
    data = reshape(data, [10000, 32, 32, 3]);
    data = permute(data, [1, 3, 2, 4]);
    
    for e_idx = 1:10

        relevant_idxes = find(labels==(e_idx-1));

        for k = 1:length(relevant_idxes)

            save_img_path = fullfile(train_save_path, num2str(e_idx));

            if ~isdir(save_img_path)
                mkdir(save_img_path);
            end
            img_path = fullfile(save_img_path, sprintf('%04d.JPEG', last_img + relevant_idxes(k)));
            
            img = squeeze(data(relevant_idxes(k), :, :, :));
            img = imresize(img, [224, 224]);   
            imwrite(img, img_path, 'jpg', 'Quality', 100);
        end 
        last_img = k;
    end
end

%% process test data
load(test_img_path);
data = reshape(data, [10000, 32, 32, 3]);
data = permute(data, [1, 3, 2, 4]);
for e_idx = 1:10

    relevant_idxes = find(labels==(e_idx-1));

    for k = 1:length(relevant_idxes)

        save_img_path = fullfile(test_save_path, num2str(e_idx));

        if ~isdir(save_img_path)
            mkdir(save_img_path);
        end
        img_path = fullfile(save_img_path, sprintf('%04d.JPEG', relevant_idxes(k)));

        img = squeeze(data(relevant_idxes(k), :, :, :));
        img = imresize(img, [224, 224]);   
        imwrite(img, img_path, 'jpg', 'Quality', 100);
    end
end
