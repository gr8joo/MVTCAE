import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.save_samples import save_generated_samples_singlegroup

def classify_cond_gen_samples(exp, labels, cond_samples, cond_mod):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clfs = exp.clfs;
    eval_labels = dict();
    avg = []
    for l, l_key in enumerate(exp.labels):
        eval_labels[l_key] = dict();
    for key in clfs.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key];
            mod_clf = clfs[key];
            attr_hat = mod_clf(mod_cond_gen);
            for l, label_str in enumerate(exp.labels):
                score = exp.eval_label(attr_hat.cpu().data.numpy(), labels, index=l);

                eval_labels[label_str][key] = score;
                if key not in cond_mod:
                    avg.append(score)
        else:
            print(str(key) + 'not existing in cond_gen_samples');
    return eval_labels, avg

'''
def classify_cond_gen_samples2(exp, labels, cond_samples, cond_mod):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clfs = exp.clfs;
    eval_labels = dict();
    for l, l_key in enumerate(exp.labels):
        eval_labels[l_key] = dict();
    pred_mods = []
    pred_mods.append( np.reshape(labels.cpu().data.numpy(), (exp.flags.batch_size,)) )
    for key in clfs.keys():
        if key in cond_samples.keys() and key not in cond_mod:
            mod_cond_gen = cond_samples[key];
            mod_clf = clfs[key];
            attr_hat = mod_clf(mod_cond_gen);
            pred_mods.append(np.argmax(attr_hat.cpu().data.numpy(), axis=-1))

    pred_mods=np.array(pred_mods)
    coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
    coherence = np.sum(coh_mods.astype(int)) / float(exp.flags.batch_size);

    return coherence;
'''



def calculate_coherence(exp, samples):
    clfs = exp.clfs;
    mods = exp.modalities;
    # TODO: make work for num samples NOT EQUAL to batch_size
    c_labels = dict();
    for j, l_key in enumerate(exp.labels):
        pred_mods = np.zeros((len(mods.keys()), exp.flags.batch_size))
        for k, m_key in enumerate(mods.keys()):
            mod = mods[m_key];
            clf_mod = clfs[mod.name];
            samples_mod = samples[mod.name];
            attr_mod = clf_mod(samples_mod);
            output_prob_mod = attr_mod.cpu().data.numpy();
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int);
            pred_mods[k, :] = pred_mod;
        coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
        coherence = np.sum(coh_mods.astype(int))/float(exp.flags.batch_size);
        c_labels[l_key] = coherence;
    return c_labels;


def test_generation(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.mm_vae;
    subsets = exp.subsets;

    gen_perf = dict();
    gen_perf = {'cond': dict(),
                'random': dict(),
                'coh': dict()}
    for j, l_key in enumerate(exp.labels):
        gen_perf['cond'][l_key] = dict();
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                gen_perf['cond'][l_key][s_key] = dict();
                for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][l_key][s_key][m_key] = [];
        gen_perf['random'][l_key] = [];

    for i in range(1, exp.num_modalities):
        gen_perf['coh'][str(i)] = []


    d_loader = DataLoader(exp.dataset_test,
                          batch_size=exp.flags.batch_size,
                          shuffle=True,
                          num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_test.__len__() /float(exp.flags.batch_size));
    cnt_s = 0;
    source_key_list = ['m1', 'm1_m2', 'm1_m2_m3', 'm1_m2_m3_m4']
    target_key = 'm0'
    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        batch_l = batch[1];
        rand_gen = mm_vae.generate();
        coherence_random = calculate_coherence(exp, rand_gen);

        for j, l_key in enumerate(exp.labels):
            gen_perf['random'][l_key].append(coherence_random[l_key]);


        if (epoch+1) % exp.flags.eval_freq_fid == 0 and (exp.flags.batch_size*iteration) < exp.flags.num_samples_fid:
            save_generated_samples_singlegroup(exp, iteration,
                                               'random',
                                               {target_key: rand_gen[target_key]});
            # save_generated_samples_singlegroup(exp, iteration,
            #                                    'real',
            #                                    batch_d);

        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
        inferred = mm_vae.inference(batch_d);
        lr_subsets = inferred['subsets'];
        cg = mm_vae.cond_generation(lr_subsets)

        for k, s_key in enumerate(cg.keys()):
            asdf = s_key.split('_')

            # if len(asdf) < exp.num_modalities:
            #     coh_cg = classify_cond_gen_samples2(exp, batch_l, cg[s_key], s_key);
            #     coh_result[len(asdf)].append(coh_cg)

            clf_cg, coh_cg = classify_cond_gen_samples(exp, batch_l, cg[s_key], s_key);
            if len(coh_cg) > 0:
                gen_perf['coh'][str(len(asdf))].extend(coh_cg)

            for j, l_key in enumerate(exp.labels):
                for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][l_key][s_key][m_key].append(clf_cg[l_key][m_key]);

            if (epoch+1) % exp.flags.eval_freq_fid == 0 and (exp.flags.batch_size*iteration) < exp.flags.num_samples_fid:
                if s_key in source_key_list:
                    save_generated_samples_singlegroup(exp, iteration,
                                                       s_key,
                                                       {target_key: cg[s_key][target_key]});



    for j, l_key in enumerate(exp.labels):
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                for l, m_key in enumerate(mods.keys()):
                    perf = exp.mean_eval_metric(gen_perf['cond'][l_key][s_key][m_key])
                    gen_perf['cond'][l_key][s_key][m_key] = perf;
        gen_perf['random'][l_key] = exp.mean_eval_metric(gen_perf['random'][l_key]);

    for i, i_key in enumerate(gen_perf['coh'].keys()):
        gen_perf['coh'][i_key] = np.mean(gen_perf['coh'][i_key])

    return gen_perf;



