import os
import sys
import argparse
import subprocess
import pdb
import time
import random
import _pickle as cPickle
import matplotlib.pyplot as plt
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from hntm import HierarchicalNeuralTopicModel
from tree import get_descendant_idxs
from evaluate import compute_hierarchical_affinity, compute_topic_specialization
from configure import get_parser
from data_structure import Instance, get_batches
from tree import get_tree_idxs

def update_checkpoint(config, checkpoint, global_step):
    checkpoint.append(config.path_model + '-%i' % global_step)
    if len(checkpoint) > config.max_to_keep:
        path_model = checkpoint.pop(0) + '.*'
        for p in glob.glob(path_model):
            os.remove(p)
    cPickle.dump(checkpoint, open(config.path_checkpoint, 'wb'))
    
def validate(sess, batches, model):
    losses = []
    ppl_list = []
    prob_topic_list = []
    n_bow_list = []
    n_topics_list = []
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch, ppls_batch, prob_topic_batch, n_bow_batch, n_topics_batch \
            = sess.run([model.loss, model.topic_loss_recon, model.topic_loss_kl, model.topic_loss_reg, model.topic_ppls, model.prob_topic, model.n_bow, model.n_topics], feed_dict = feed_dict)
        losses += [[loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch]]
        ppl_list += list(ppls_batch)
        prob_topic_list.append(prob_topic_batch)
        n_bow_list.append(n_bow_batch)
        n_topics_list.append(n_topics_batch)
    loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean = np.mean(losses, 0)
    ppl_mean = np.exp(np.mean(ppl_list))
    
    probs_topic = np.concatenate(prob_topic_list, 0)
    
    n_bow = np.concatenate(n_bow_list, 0)
    n_topics = np.concatenate(n_topics_list, 0)
    probs_topic_mean = np.sum(n_topics, 0) / np.sum(n_bow)
    
    return loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean, ppl_mean, probs_topic_mean

def print_topic_sample(sess, model, topic_prob_topic=None, recur_prob_topic=None, topic_freq_tokens=None, parent_idx=0, depth=0):
    if depth == 0: # print root
        assert len(topic_prob_topic) == len(recur_prob_topic) == len(topic_freq_tokens)
        freq_tokens = topic_freq_tokens[parent_idx]
        recur_topic = recur_prob_topic[parent_idx]
        prob_topic = topic_prob_topic[parent_idx]
        print(parent_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        freq_tokens = topic_freq_tokens[child_idx]
        recur_topic = recur_prob_topic[child_idx]
        prob_topic = topic_prob_topic[child_idx]
        print('  '*depth, child_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
        
        if child_idx in model.tree_idxs: 
            print_topic_sample(sess, model, topic_prob_topic=topic_prob_topic, recur_prob_topic=recur_prob_topic, topic_freq_tokens=topic_freq_tokens, parent_idx=child_idx, depth=depth)
            
if __name__ == '__main__':
    # load config
    parser = get_parser()
    config = parser.parse_args()

    config.tree_idxs = get_tree_idxs(config.tree)
    config.path_data = os.path.join('data', config.data, 'instances.pkl')
    config.dir_model = os.path.join('model', config.data, 'checkpoint_tmp')
    config.dir_corpus = os.path.join('corpus', config.data)
    config.path_model = os.path.join(config.dir_model, 'model') 
    config.path_config = config.path_model + '-%i.config'
    config.path_log = os.path.join(config.dir_model, 'log')
    config.path_checkpoint = os.path.join(config.dir_model, 'checkpoint')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    np.random.seed(config.seed)
    random.seed(config.seed)

    # load data
    instances_train, instances_dev, _, word_to_idx, idx_to_word, bow_idxs = cPickle.load(open(config.path_data,'rb'))
    train_batches = get_batches(instances_train, config.batch_size)
    dev_batches = get_batches(instances_dev, config.batch_size)
    config.dim_bow = len(bow_idxs)
    
    # initialize log
    checkpoint = []
    losses_train = []
    ppls_train = []
    ppl_min = np.inf
    epoch = 0
    train_batches = get_batches(instances_train, config.batch_size, iterator=True)

    log_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                        list(zip(*[['','','','TRAIN:','','','','','VALID:','','','','', 'SPEC:', '', '', 'HIER:', ''],
                                ['Time','Ep','Ct','LOSS','PPL','NLL','KL','REG','LOSS','PPL','NLL','KL','REG', '1', '2', '3', 'CHILD', 'OTHER']]))))

    cmd_rm = 'rm -r %s' % config.dir_model
    res = subprocess.call(cmd_rm.split())
    cmd_mk = 'mkdir %s' % config.dir_model
    res = subprocess.call(cmd_mk.split())
    
    # initialize model
    if 'sess' in globals(): sess.close()
    model = HierarchicalNeuralTopicModel(config)
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=config.max_to_keep)
    update_tree_flg = False
    
    # train model
    time_start = time.time()
    while epoch < config.n_epochs:
        # train
        for ct, batch in enumerate(train_batches):
            feed_dict = model.get_feed_dict(batch)
            _, loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch, ppls_batch, global_step_log = \
            sess.run([model.opt, model.loss, model.topic_loss_recon, model.topic_loss_kl, model.topic_loss_reg, model.topic_ppls, tf.train.get_global_step()], feed_dict = feed_dict)

            losses_train += [[loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch]]
            ppls_train += list(ppls_batch)

            if global_step_log % config.log_period == 0:
                # validate
                loss_train, topic_loss_recon_train, topic_loss_kl_train, topic_loss_reg_train = np.mean(losses_train, 0)
                ppl_train = np.exp(np.mean(ppls_train))
                loss_dev, topic_loss_recon_dev, topic_loss_kl_dev, topic_loss_reg_dev, ppl_dev, probs_topic_dev = validate(sess, dev_batches, model)

                # save model
                if ppl_dev < ppl_min:
                    ppl_min = ppl_dev
                    saver.save(sess, config.path_model, global_step=global_step_log)
                    cPickle.dump(config, open(config.path_config % global_step_log, 'wb'))
                    update_checkpoint(config, checkpoint, global_step_log)

                # visualize topic
                topics_freq_indices = np.argsort(sess.run(model.topic_bow), 1)[:, ::-1][:, :config.n_freq]
                topics_freq_idxs = bow_idxs[topics_freq_indices]
                topic_freq_tokens = {topic_idx: [idx_to_word[idx] for idx in topic_freq_idxs] for topic_idx, topic_freq_idxs in zip(model.topic_idxs, topics_freq_idxs)}
                topic_prob_topic = {topic_idx: prob_topic for topic_idx, prob_topic in zip(model.topic_idxs, probs_topic_dev)}
                descendant_idxs = {parent_idx: get_descendant_idxs(model, parent_idx) for parent_idx in model.topic_idxs}
                recur_prob_topic = {parent_idx: np.sum([topic_prob_topic[child_idx] for child_idx in recur_child_idxs]) for parent_idx, recur_child_idxs in descendant_idxs.items()}

                depth_specs = compute_topic_specialization(sess, model, instances_dev)
                hierarchical_affinities = compute_hierarchical_affinity(sess, model)

                # save log
                time_log = int(time.time() - time_start)
                log_series = pd.Series([time_log, epoch, ct, \
                        '%.2f'%loss_train, '%.0f'%ppl_train, '%.2f'%topic_loss_recon_train, '%.2f'%topic_loss_kl_train, '%.2f'%topic_loss_reg_train, \
                        '%.2f'%loss_dev, '%.0f'%ppl_dev, '%.2f'%topic_loss_recon_dev, '%.2f'%topic_loss_kl_dev, '%.2f'%topic_loss_reg_dev, \
                        '%.2f'%depth_specs[1], '%.2f'%depth_specs[2], '%.2f'%depth_specs[3], \
                        '%.2f'%hierarchical_affinities[0], '%.2f'%hierarchical_affinities[1]],
                        index=log_df.columns)
                log_df.loc[global_step_log] = log_series
                print(log_series)
                cPickle.dump(log_df, open(os.path.join(config.path_log), 'wb'))
                print_topic_sample(sess, model, topic_prob_topic=topic_prob_topic, recur_prob_topic=recur_prob_topic, topic_freq_tokens=topic_freq_tokens)

                # update tree
                if not config.static:
                    config.tree_idxs, update_tree_flg = model.update_tree(topic_prob_topic, recur_prob_topic)
                    if update_tree_flg:
                        print(config.tree_idxs)
                        name_variables = {tensor.name: variable for tensor, variable in zip(tf.global_variables(), sess.run(tf.global_variables()))} # store paremeters
                        if 'sess' in globals(): sess.close()
                        model = HierarchicalNeuralTopicModel(config)
                        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
                        name_tensors = {tensor.name: tensor for tensor in tf.global_variables()}
                        sess.run([name_tensors[name].assign(variable) for name, variable in name_variables.items()]) # restore parameters
                        saver = tf.train.Saver(max_to_keep=1)

                time_start = time.time()

        train_batches = get_batches(instances_train, config.batch_size, iterator=True)
        epoch += 1