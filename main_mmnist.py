import sys
import os
import json

import torch

from run_epochs import run_epochs

from utils.filehandling import create_dir_structure
from mmnist.flags import parser
from mmnist.experiment import MMNISTExperiment

if __name__ == '__main__':

    ############### SEED has to be set in run_epochs.py ###############
    ### Changing SEED here only affects the name of the directory. ###
    SEED = 0

    METHOD = 'tc'#'joint_elbo', 'jsd', 'moe', 'poe'
    TC_RATIO = 5.0/6.0
    BETA = 2.5
    FACTORIZED = False

    BASE_DIR = "."
    DIR_EXPERIMENT_BASE = BASE_DIR + '/exp_poly/' + METHOD + '/' + 'BETA' + str(BETA)
    DIR_EXPERIMENT_BASE += '_SEED' + str(SEED)


    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

    FLAGS.datapaths = './data'
    FLAGS.unimodal_datapaths_train= [FLAGS.datapaths + '/MMNIST/train/m'+str(i)+'.pt' for i in range(5)]
    FLAGS.unimodal_datapaths_test = [FLAGS.datapaths + '/MMNIST/test/m'+str(i)+'.pt' for i in range(5)]
    FLAGS.unimodal_labelpaths_train= FLAGS.datapaths + '/MMNIST/train/labels.pt'
    FLAGS.unimodal_labelpaths_test = FLAGS.datapaths + '/MMNIST/test/labels.pt'
    FLAGS.pretrained_classifier_paths = [FLAGS.datapaths + '/clf/pretrained_img_to_digit_clf_m' + str(i) for i in range(5)]
    FLAGS.inception_state_dict = FLAGS.datapaths + '/inception_state_dict.pth'
    FLAGS.class_dim = 512
    FLAGS.eval_freq = 25
    FLAGS.eval_freq_fid = 300
    FLAGS.end_epoch = FLAGS.eval_freq_fid + FLAGS.eval_freq  # To ensure that all the records at 300-th epoch are properly saved in tensorboard.

    # SEED has to be set in run_epochs.py
    FLAGS.seed = SEED
    FLAGS.method = METHOD
    FLAGS.beta = BETA
    FLAGS.tc_ratio = TC_RATIO
    FLAGS.dir_experiment = DIR_EXPERIMENT_BASE
    FLAGS.dir_fid = DIR_EXPERIMENT_BASE


    if FLAGS.method == 'poe':
        FLAGS.modality_poe = True
        FLAGS.poe_unimodal_elbos = True
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe = True
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd = True
    elif FLAGS.method == 'joint_elbo':
        FLAGS.joint_elbo = True
    elif FLAGS.method == 'tc':
        FLAGS.modality_ivw = True;
    else:
        print('method implemented...exit!')
        sys.exit()
    print(FLAGS.modality_poe)
    print(FLAGS.modality_moe)
    print(FLAGS.modality_jsd)
    print(FLAGS.joint_elbo)
    print(FLAGS.modality_ivw)

    # postprocess flags
    assert len(FLAGS.unimodal_datapaths_train) == len(FLAGS.unimodal_datapaths_test)
    FLAGS.num_mods = len(FLAGS.unimodal_datapaths_train)  # set number of modalities dynamically
    if FLAGS.div_weight_uniform_content is None:
        FLAGS.div_weight_uniform_content = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content]
    if FLAGS.div_weight is None:
        FLAGS.div_weight = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities.extend([FLAGS.div_weight for _ in range(FLAGS.num_mods)])
    print("alpha_modalities:", FLAGS.alpha_modalities)
    create_dir_structure(FLAGS)

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mst = MMNISTExperiment(FLAGS, alphabet)
    mst.set_optimizer()

    print(FLAGS)
    run_epochs(mst)
