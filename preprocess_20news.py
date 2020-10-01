import os
import re
import random
import argparse
import _pickle as cPickle
from collections import OrderedDict, defaultdict, Counter

import numpy as np
import pandas as pd
from data_structure import Instance

random.seed(1234)

def get_df(path_data):
    data_dict = {}
    docs = np.load(path_data, allow_pickle=True, encoding='bytes')
    for token_idxs in docs:
        data_dict['token_idxs'] = token_idxs
        data_dict['doc_l'] = len(token_idxs)
    data_df = pd.DataFrame(data_dict)
    return data_df

def prepare_instances(path_data, bow_idxs):
    instances = []
    docs = np.load(path_data, allow_pickle=True, encoding='bytes')
    for idx_doc, token_idxs in enumerate(docs):
        if len(token_idxs) == 0: continue
        instance = Instance()
        instance.idx = idx_doc
        instance.token_idxs = token_idxs
        instance.doc_l = len(token_idxs)
        token_idx_cnt = Counter(token_idxs)
        instance.bow = np.array([token_idx_cnt[bow_idx] for bow_idx in bow_idxs])
        if not (sum(token_idx_cnt.values()) == np.sum(instance.bow) == len(instance.token_idxs)):
            print('skip: %i' % idx_doc)
        instances.append(instance)
    return instances

def write_corpus(config, idx_to_word):
    data_train = np.load(config.path_train, allow_pickle=True, encoding='bytes')
    data_test = np.load(config.path_test, allow_pickle=True, encoding='bytes')
    data = np.concatenate([data_train, data_test])
    
    n_doc = 1000
    docs = []
    for i, instance in enumerate(data):
        doc = ' '.join([idx_to_word[word_idx] for word_idx in instance])
        docs.append(doc)
        if (i+1) % n_doc == 0:
            fname = '20news.%i' % (i // n_doc)
            with open(os.path.join(config.dir_corpus, fname), 'w', encoding='utf-8') as f:
                f.write('\n'.join(docs))
                docs = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir_data', type=str, default='data/20news')
    parser.add_argument('-path_split', type=str, default='data/20news/item_idxs_split.pkl')
    parser.add_argument('-path_output', type=str, default='data/20news/instances.pkl')
    parser.add_argument('-dir_corpus', type=str, default='corpus/20news')
    config = parser.parse_args('')
    config.path_train = os.path.join(config.dir_data, 'train.txt.npy')
    config.path_test = os.path.join(config.dir_data, 'test.txt.npy')
    config.path_vocab = os.path.join(config.dir_data, 'vocab.pkl')
    
    word_to_idx = cPickle.load(open(config.path_vocab, 'rb'))
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    bow_idxs = np.unique(list(word_to_idx.values()))

    instances_train_dev = prepare_instances(config.path_train, bow_idxs)
    instances_test = prepare_instances(config.path_test, bow_idxs)
    item_idxs_train, item_idxs_dev = cPickle.load(open(config.path_split, 'rb'))
    instances_train = list(np.array(instances_train_dev)[item_idxs_train])
    instances_dev = list(np.array(instances_train_dev)[item_idxs_dev])
    assert len(instances_train) + len(instances_dev) == len(instances_train_dev)

    print('saving preprocessed instances...')
    cPickle.dump((instances_train, instances_dev, instances_test, word_to_idx, idx_to_word, bow_idxs), open(config.path_output, 'wb'))