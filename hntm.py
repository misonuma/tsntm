#coding: utf-8

from collections import defaultdict

import copy
import numpy as np
import tensorflow as tf

from nn import doubly_rnn, rnn, tsbp, sbp
from components import tf_log, sample_latents, compute_kl_losses, softmax_with_temperature
from tree import get_topic_idxs, get_child_to_parent_idxs, get_depth, get_ancestor_idxs, get_descendant_idxs

class HierarchicalNeuralTopicModel():
    def __init__(self, config):
        self.config = config
        
        self.t_variables = {}
        self.tree_idxs = config.tree_idxs
        self.topic_idxs = get_topic_idxs(self.tree_idxs)
        self.child_to_parent_idxs = get_child_to_parent_idxs(self.tree_idxs)
        self.tree_depth = get_depth(self.tree_idxs)
        self.n_depth = max(self.tree_depth.values())
        
        self.build()
        
    def build(self):
        def get_prob_topic(tree_prob_leaf, prob_depth):
            tree_prob_topic = defaultdict(float)
            leaf_ancestor_idxs = {leaf_idx: get_ancestor_idxs(leaf_idx, self.child_to_parent_idxs) for leaf_idx in tree_prob_leaf}
            for leaf_idx, ancestor_idxs in leaf_ancestor_idxs.items():
                prob_leaf = tree_prob_leaf[leaf_idx]
                for i, ancestor_idx in enumerate(ancestor_idxs):
                    prob_ancestor = prob_leaf * tf.expand_dims(prob_depth[:, i], -1)
                    tree_prob_topic[ancestor_idx] += prob_ancestor
            prob_topic = tf.concat([tree_prob_topic[topic_idx] for topic_idx in self.topic_idxs], -1)
            return prob_topic     
        
        def get_tree_topic_bow(tree_topic_embeddings):
            tree_topic_bow = {}
            for topic_idx, depth in self.tree_depth.items():
                topic_embedding = tree_topic_embeddings[topic_idx]
                temperature = tf.constant(self.config.depth_temperature ** (1./depth), dtype=tf.float32)
                logits = tf.matmul(topic_embedding, self.bow_embeddings, transpose_b=True)
                tree_topic_bow[topic_idx] = softmax_with_temperature(logits, axis=-1, temperature=temperature)
                
            return tree_topic_bow
        
        def get_topic_loss_reg(tree_topic_embeddings):
            def get_tree_mask_reg(all_child_idxs):        
                tree_mask_reg = np.zeros([len(all_child_idxs), len(all_child_idxs)], dtype=np.float32)
                for parent_idx, child_idxs in self.tree_idxs.items():
                    neighbor_idxs = child_idxs
                    for neighbor_idx1 in neighbor_idxs:
                        for neighbor_idx2 in neighbor_idxs:
                            neighbor_index1 = all_child_idxs.index(neighbor_idx1)
                            neighbor_index2 = all_child_idxs.index(neighbor_idx2)
                            tree_mask_reg[neighbor_index1, neighbor_index2] = tree_mask_reg[neighbor_index2, neighbor_index1] = 1.
                return tree_mask_reg
            
            all_child_idxs = list(self.child_to_parent_idxs.keys())
            self.diff_topic_embeddings = tf.concat([tree_topic_embeddings[child_idx] - tree_topic_embeddings[self.child_to_parent_idxs[child_idx]] for child_idx in all_child_idxs], axis=0)
            diff_topic_embeddings_norm = self.diff_topic_embeddings / tf.norm(self.diff_topic_embeddings, axis=1, keepdims=True)
            self.topic_dots = tf.clip_by_value(tf.matmul(diff_topic_embeddings_norm, tf.transpose(diff_topic_embeddings_norm)), -1., 1.)        

            self.tree_mask_reg = get_tree_mask_reg(all_child_idxs)
            self.topic_losses_reg = tf.square(self.topic_dots - tf.eye(len(all_child_idxs))) * self.tree_mask_reg
            self.topic_loss_reg = tf.reduce_sum(self.topic_losses_reg) / tf.reduce_sum(self.tree_mask_reg)
            return self.topic_loss_reg
           
        # -------------- Build Model --------------
        tf.reset_default_graph()
        
        tf.set_random_seed(self.config.seed)
        
        self.t_variables['bow'] = tf.placeholder(tf.float32, [None, self.config.dim_bow])
        self.t_variables['keep_prob'] = tf.placeholder(tf.float32)
        
        # encode bow
        with tf.variable_scope('topic/enc', reuse=False):
            hidden_bow_ = tf.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.tanh, name='hidden_bow')(self.t_variables['bow'])
            hidden_bow = tf.layers.Dropout(self.t_variables['keep_prob'])(hidden_bow_)
            means_bow = tf.layers.Dense(units=self.config.dim_latent_bow, name='mean_bow')(hidden_bow)
            logvars_bow = tf.layers.Dense(units=self.config.dim_latent_bow, kernel_initializer=tf.constant_initializer(0), bias_initializer=tf.constant_initializer(0), name='logvar_bow')(hidden_bow)
            latents_bow = sample_latents(means_bow, logvars_bow) # sample latent vectors
            prob_layer = lambda h: tf.nn.sigmoid(tf.matmul(latents_bow, h, transpose_b=True))

            tree_sticks_topic, tree_states_sticks_topic = doubly_rnn(self.config.dim_latent_bow, self.tree_idxs, output_layer=prob_layer, cell=self.config.cell, name='sticks_topic')
            self.tree_prob_leaf = tsbp(tree_sticks_topic, self.tree_idxs)
            
            sticks_depth, _ = rnn(self.config.dim_latent_bow, self.n_depth, output_layer=prob_layer, cell=self.config.cell, name='prob_depth')
            self.prob_depth = sbp(sticks_depth, self.n_depth)

            self.prob_topic = get_prob_topic(self.tree_prob_leaf, self.prob_depth)# n_batch x n_topic

        # decode bow
        with tf.variable_scope('shared', reuse=False):
            self.bow_embeddings = tf.get_variable('emb', [self.config.dim_bow, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of vocab

        with tf.variable_scope('topic/dec', reuse=False):
            emb_layer = lambda h: tf.layers.Dense(units=self.config.dim_emb, name='output')(tf.nn.tanh(h))
            self.tree_topic_embeddings, tree_states_topic_embeddings = doubly_rnn(self.config.dim_emb, self.tree_idxs, output_layer=emb_layer, cell=self.config.cell, name='emb_topic')

            self.tree_topic_bow = get_tree_topic_bow(self.tree_topic_embeddings) # bow vectors for each topic

            self.topic_bow = tf.concat([self.tree_topic_bow[topic_idx] for topic_idx in self.topic_idxs], 0) # KxV
            self.logits_bow = tf_log(tf.matmul(self.prob_topic, self.topic_bow)) # predicted bow distribution N_Batch x  V
            
        # define losses
        self.topic_losses_recon = -tf.reduce_sum(tf.multiply(self.t_variables['bow'], self.logits_bow), 1)
        self.topic_loss_recon = tf.reduce_mean(self.topic_losses_recon) # negative log likelihood of each words

        self.topic_losses_kl = compute_kl_losses(means_bow, logvars_bow) # KL divergence b/w latent dist & gaussian std
        self.topic_loss_kl = tf.reduce_mean(self.topic_losses_kl, 0) #mean of kl_losses over batches        
        
        self.topic_embeddings = tf.concat([self.tree_topic_embeddings[topic_idx] for topic_idx in self.topic_idxs], 0) # temporary
        self.topic_loss_reg = get_topic_loss_reg(self.tree_topic_embeddings)

        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.loss = self.topic_loss_recon + self.topic_loss_kl + self.config.reg * self.topic_loss_reg

        # define optimizer
        if self.config.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.config.lr)

        self.grad_vars = optimizer.compute_gradients(self.loss)
        self.clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in self.grad_vars]
        self.opt = optimizer.apply_gradients(self.clipped_grad_vars, global_step=self.global_step)

        # monitor
        self.n_bow = tf.reduce_sum(self.t_variables['bow'], 1)
        self.topic_ppls = tf.divide(self.topic_losses_recon + self.topic_losses_kl, tf.maximum(1e-5, self.n_bow))
    
        # growth criteria
        self.n_topics = tf.multiply(tf.expand_dims(self.n_bow, -1), self.prob_topic)
        
        self.arcs_bow = tf.acos(tf.matmul(tf.linalg.l2_normalize(self.bow_embeddings, axis=-1), tf.linalg.l2_normalize(self.topic_embeddings, axis=-1), transpose_b=True)) # n_vocab x n_topic
        self.rads_bow = tf.multiply(tf.matmul(self.t_variables['bow'], self.arcs_bow), self.prob_topic) # n_batch x n_topic
    
    def get_feed_dict(self, batch, mode='train'):
        bow = np.array([instance.bow for instance in batch]).astype(np.float32)
        keep_prob = self.config.keep_prob if mode == 'train' else 1.0
        feed_dict = {
                    self.t_variables['bow']: bow, 
                    self.t_variables['keep_prob']: keep_prob
        }
        return  feed_dict
    
    def update_tree(self, topic_prob_topic, recur_prob_topic):
        assert len(self.topic_idxs) == len(recur_prob_topic) == len(topic_prob_topic)
        update_tree_flg = False

        def add_topic(topic_idx, tree_idxs):
            if topic_idx in tree_idxs:
                child_idx = min([10*topic_idx+i for i in range(1, 10) if 10*topic_idx+i not in tree_idxs[topic_idx]])
                tree_idxs[topic_idx].append(child_idx)        
            else:
                child_idx = 10*topic_idx+1
                tree_idxs[topic_idx] = [10*topic_idx+1]
            return tree_idxs, child_idx

        added_tree_idxs = copy.deepcopy(self.tree_idxs)
        for parent_idx, child_idxs in self.tree_idxs.items():
            prob_topic = topic_prob_topic[parent_idx]
            if prob_topic > self.config.add_threshold:
                update_tree_flg = True
                for depth in range(self.tree_depth[parent_idx], self.n_depth):
                    added_tree_idxs, parent_idx = add_topic(parent_idx, added_tree_idxs)

        def remove_topic(parent_idx, child_idx, tree_idxs):
            if parent_idx in tree_idxs:
                tree_idxs[parent_idx].remove(child_idx)
                if child_idx in tree_idxs:
                    tree_idxs.pop(child_idx)
            return tree_idxs

        removed_tree_idxs = copy.deepcopy(added_tree_idxs)
        for parent_idx, child_idxs in self.tree_idxs.items():
            probs_child = np.array([recur_prob_topic[child_idx] for child_idx in child_idxs])
            for prob_child, child_idx in zip(probs_child, child_idxs):
                if prob_child < self.config.remove_threshold:
                    update_tree_flg = True
                    removed_tree_idxs = remove_topic(parent_idx, child_idx, removed_tree_idxs)
                    if parent_idx in removed_tree_idxs:
                        if len(removed_tree_idxs[parent_idx]) == 0:
                            ancestor_idx = self.child_to_parent_idxs[parent_idx]
                            removed_tree_idxs = remove_topic(ancestor_idx, parent_idx, removed_tree_idxs)
        return removed_tree_idxs, update_tree_flg    
