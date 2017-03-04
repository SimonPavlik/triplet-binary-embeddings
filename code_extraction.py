import yaml
import numpy as np
import scipy.io as sio
import os

# os.environ["THEANO_FLAGS"] = "floatX=float32,device=gpu0,lib.cnmem=1"


from lib.tools import (save_weights, load_weights, load_weights_finetune, load_feature_extraction_weights,
                   initialize_weights, save_momentums, load_momentums)
from train_funcs import (unpack_configs, proc_configs, get_prediction_labels)


def code_extraction(config):

    # UNPACK CONFIGS
    (train_filenames, val_filenames, img_mean) = unpack_configs(config)

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['gpu'])

    import theano
    theano.config.on_unused_input = 'warn'
    import theano.tensor as T

    from lib.multilabel_layers import DropoutLayer
    from multilabel_net import CNN_model, compile_models

    # load hash_step1_bits
    group_idx = sio.loadmat('./step1/temp/group_idx.mat')
    group_idx = group_idx['group_idx']
    group_idx = group_idx[0][0]
    code_per_group = 8

    bits_idxes = range((group_idx - 1) * code_per_group)

    config['output_num'] = len(bits_idxes)


    model = CNN_model(config)
    batch_size = model.batch_size
    layers = model.layers

    n_train_batches = len(train_filenames)
    n_val_batches = len(val_filenames)

    ## COMPILE FUNCTIONS ##
    (_, _, predict_model, _, _, shared_x, _, _) = compile_models(model, config)

    load_weights_epoch = config['load_weights_epoch']

    train_predicted_code = None
    val_predicted_code = None

    load_weights_dir = config['weights_dir']

    load_weights(layers, load_weights_dir, load_weights_epoch)

    code_save_dir = config['code_save_dir']

    DropoutLayer.SetDropoutOff()

    for minibatch_index in range(n_train_batches):

        label = get_prediction_labels(predict_model, shared_x, minibatch_index, train_filenames, img_mean)

        if train_predicted_code is None:
            train_predicted_code = label[0]
        else:
            train_predicted_code = np.vstack((train_predicted_code, label[0]))

    database_code = {'database_code': train_predicted_code}
    sio.savemat(code_save_dir + 'database_code.mat', database_code)

    for minibatch_index in range(n_val_batches):

        label = get_prediction_labels(predict_model, shared_x, minibatch_index, val_filenames, img_mean)
        if val_predicted_code is None:
            val_predicted_code = label[0]
        else:
            val_predicted_code = np.vstack((val_predicted_code, label[0]))

    test_code = {'test_code': val_predicted_code}
    sio.savemat(code_save_dir + 'test_code.mat', test_code)

    print('code extraction complete.')


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)

    config = proc_configs(config)

    code_extraction(config)
