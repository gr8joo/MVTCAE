


class TBLogger():
    def __init__(self, name, writer):
        self.name = name;
        self.writer = writer;
        self.training_prefix = 'train';
        self.testing_prefix = 'test';
        self.step = 0;


    def write_log_probs(self, name, log_probs):
        # self.writer.add_scalars('%s/LogProb' % name,
        #                         log_probs,
        #                         self.step)
        for k, key in enumerate( sorted(log_probs.keys()) ):
            self.writer.add_scalar(name + '/RecLoss/' + key,
                                   log_probs[key],
                                   self.step)


    def write_klds(self, name, klds):
        # self.writer.add_scalars('%s/KLD' % name,
        #                         klds,
        #                         self.step)
        for k, key in enumerate( sorted(klds.keys()) ):
            self.writer.add_scalar(name + '/KLD/' + key,
                                   klds[key],
                                   self.step)


    def write_group_div(self, name, group_div):
        # self.writer.add_scalars('%s/group_divergence' % name,
        #                         {'group_div': group_div.item()},
        #                         self.step)

        self.writer.add_scalar(name + '/group_divergence/' + 'group_div',
                               group_div.item(),
                               self.step)


    def write_latent_distr(self, name, latents):
        l_mods = latents['modalities'];
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                # self.writer.add_scalars('%s/mu' % name,
                #                         {key: l_mods[key][0].mean().item()},
                #                         self.step)
                self.writer.add_scalar(name + '/mu/' + key,
                                       l_mods[key][0].mean().item(),
                                       self.step)
            if not l_mods[key][1] is None:
                # self.writer.add_scalars('%s/logvar' % name,
                #                         {key: l_mods[key][1].mean().item()},
                #                         self.step)
                self.writer.add_scalar(name + '/logvar/' + key,
                                       l_mods[key][1].mean().item(),
                                       self.step)


    def write_lr_eval(self, lr_eval):
        for l, l_key in enumerate( sorted(lr_eval.keys()) ):
            # self.writer.add_scalars('Latent Representation/%s'%(l_key),
            #                         lr_eval[l_key],
            #                         self.step)
            summary = {'1':[],
                       '2':[],
                       '3':[],
                       '4':[],
                       '5':[]}
            for k, key in enumerate( sorted(lr_eval[l_key].keys()) ):
                self.writer.add_scalar('Latent Representation/' + l_key + '/' + key,
                                       lr_eval[l_key][key],
                                       self.step)
                summary[ str(len(key.split('_'))) ].append(lr_eval[l_key][key])

            for k, key in enumerate( summary.keys() ):
                if len(summary[key]) > 0:
                    self.writer.add_scalar('Classification/given' + key,
                                           sum(summary[key]) / len(summary[key]),#np.mean(summary[key]),
                                           self.step)


    def write_coherence_logs(self, gen_eval):
        for l, l_key in enumerate( sorted(gen_eval['cond'].keys()) ):
            for s, s_key in enumerate( sorted(gen_eval['cond'][l_key].keys()) ):
                # self.writer.add_scalars('Generation/%s/%s' %
                #                         (l_key, s_key),
                #                         gen_eval['cond'][l_key][s_key],
                #                         self.step)
                for k, key in enumerate( sorted(gen_eval['cond'][l_key][s_key].keys()) ):
                    if key not in s_key:
                        self.writer.add_scalar('Generation/' + l_key + '/' + s_key + '/' + key,
                                               gen_eval['cond'][l_key][s_key][key],
                                               self.step)

        # self.writer.add_scalars('Generation/Random',
        #                         gen_eval['random'],
        #                         self.step)
        for k, key in enumerate( sorted(gen_eval['random'].keys()) ):
            self.writer.add_scalar('Coherence/Random/' + key,
                                   gen_eval['random'][key],
                                   self.step)

        for k, key in enumerate( sorted(gen_eval['coh'].keys()) ):
            self.writer.add_scalar('Coherence/given' + key,
                                   gen_eval['coh'][key],
                                   self.step)


    def write_lhood_logs(self, lhoods):
        for k, key in enumerate( sorted(lhoods.keys()) ):
            # self.writer.add_scalars('Likelihoods/%s'%
            #                         (key),
            #                         lhoods[key],
            #                         self.step)
            for l, l_key in enumerate( sorted(lhoods['key']['l_key'].keys()) ):
                self.writer.add_scalar('Likelihoods/' + key + '/' + l_key,
                                       lhoods[key][l_key],
                                       self.step)

    def write_prd_scores(self, prd_scores):
        # self.writer.add_scalars('PRD',
        #                         prd_scores,
        #                         self.step)
        for k, key in enumerate( sorted(prd_scores.keys()) ):
            self.writer.add_scalar('PRD/' + key,
                                   prd_scores[key],
                                   self.step)


    def write_plots(self, plots, epoch):
        for k, p_key in enumerate(plots.keys()):
            ps = plots[p_key];
            for l, name in enumerate(ps.keys()):
                fig = ps[name];
                self.writer.add_image(p_key + '_' + name,
                                      fig,
                                      epoch,
                                      dataformats="HWC");



    def add_basic_logs(self, name, results, loss, log_probs, klds):
        # self.writer.add_scalars('%s/Loss' % name,
        #                         {'loss': loss.data.item()},
        #                         self.step)
        self.writer.add_scalar(name + '/Loss/' + 'loss',
                               loss.data.item(),
                               self.step)
        self.write_log_probs(name, log_probs);
        self.write_klds(name, klds);
        self.write_group_div(name, results['joint_divergence']);
        self.write_latent_distr(name, results['latents']);


    def write_training_logs(self, results, loss, log_probs, klds):
        self.add_basic_logs(self.training_prefix, results, loss, log_probs, klds);
        self.step += 1;


    def write_testing_logs(self, results, loss, log_probs, klds):
        self.add_basic_logs(self.testing_prefix, results, loss, log_probs, klds);
        self.step += 1;





