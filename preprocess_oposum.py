import os
import re
import random
import argparse
import _pickle as cPickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from data_structure import Instance

random.seed(1234)

def get_data_df(config):
    def preprocess(line):
        line = re.sub(r'EDU_BREAK ', ' ', line)
        line = re.sub(r'[0-9]+.[0-9]+|[0-9]+,[0-9]+|[0-9]+', '#', line)
        line = line.lower()
        return line
    
    # load corpus
    with open(config.path_data, 'r') as f:
        txt = f.read()
    docs = txt.split('\n\n')[:-1]
    docs_lines = [doc.split('\n') for doc in docs]
    data_dict_list = [{'review_idx': doc_lines[0].split(' ')[0], 'score': doc_lines[0].split(' ')[1], 'lines': [preprocess(line) for line in doc_lines[1:]]} for doc_lines in docs_lines]
    data_df = pd.DataFrame(data_dict_list)
    
    # preprocess tokens
    get_tokens = lambda lines: [word_tokenize(line.lower())[:-1] for line in lines]
    data_df['tokens'] = data_df['lines'].apply(get_tokens)
    filter_tokens = lambda tokens: [line_tokens for line_tokens in tokens if len(line_tokens) > 2]
    data_df['tokens'] = data_df['tokens'].apply(filter_tokens)
    data_df['doc_l'] = data_df['tokens'].apply(lambda tokens: len(tokens))
    data_df['max_sent_l'] = data_df['tokens'].apply(lambda tokens: max([len(line) for line in tokens]))
    data_df['item_idx'] = data_df['review_idx'].apply(lambda review_idx: review_idx.split('-')[0])
    data_df['doc'] = data_df['lines'].apply(lambda lines: ' '.join(lines))
    
    filtered_data_df = data_df[(data_df['doc_l'] <= config.max_doc_l) & (data_df['max_sent_l'] <= config.max_sent_l)]
    return filtered_data_df

def split_data_df(config, data_df):
    item_idxs_split = cPickle.load(open(config.path_split, 'rb'))
    test_df = data_df[data_df['item_idx'].apply(lambda item_idx: item_idx in item_idxs_split['test'])]
    dev_df = data_df[data_df['item_idx'].apply(lambda item_idx: item_idx in item_idxs_split['dev'])]
    train_df = data_df[data_df['item_idx'].apply(lambda item_idx: item_idx in item_idxs_split['train'])]

    return train_df, dev_df, test_df

def get_word_cnt_dict(train_df, min_tf=None):
    # create vocab of words
    word_cnt_dict = defaultdict(int)
    
    tokens_list = []
    for doc in train_df.tokens:
        tokens_list.extend(doc)
    
    for tokens in tokens_list:
        for word in tokens:
            word_cnt_dict[word] += 1
    word_cnt_dict = sorted(word_cnt_dict.items(), key=lambda x: x[1])[::-1]
    
    if type(min_tf) is int:
        word_cnt_dict = list(filter(lambda x: x[1] >= min_tf, word_cnt_dict))
    elif type(min_tf) is float:
        word_cnt_dict = word_cnt_dict[:int(min_tf*len(word_cnt_dict))]
    return word_cnt_dict

def set_bow(config, train_df, dev_df, test_df, word_to_idx):
    def get_stop_words(config, train_df):
        stop_word_cnt_dict = get_word_cnt_dict(train_df, min_tf=config.min_df_stop)

        with open(config.path_stopwords, 'r') as f:
            stop_words_mallet = [w.replace('\n', '') for w in f.readlines()]

        stop_words = stop_words_mallet + [w_cnt[0] for w_cnt in stop_word_cnt_dict] + ['-lrb-', '-rrb-', '``', "'", ';', ':', '&', '!', '$', '-', '...', '--',"'m", "'ve", "'d", "'ll", "'re"]     

        return stop_words    
    
    stop_words = get_stop_words(config, train_df)
    
    train_corpus = list(train_df.doc)
    vectorizer = TfidfVectorizer(min_df=config.min_df, max_df=config.max_df, stop_words=stop_words, tokenizer=word_tokenize, norm=None, use_idf=False, dtype=np.float32)
    train_bow_list = vectorizer.fit_transform(train_corpus)
    bow_tokens = vectorizer.get_feature_names()
    print('Number of words in vocabulary:', len(bow_tokens))
    assert len(train_df) == len(train_bow_list.toarray())
    assert all([word in word_to_idx for word in bow_tokens])

    dev_corpus = list(dev_df.doc)
    test_corpus = list(test_df.doc)
    dev_bow_list = vectorizer.transform(dev_corpus)
    test_bow_list = vectorizer.transform(test_corpus)
    bow_idxs = np.array([word_to_idx[token] for token in bow_tokens])
    
    train_df['bow'] = list(train_bow_list.toarray())
    dev_df['bow'] = list(dev_bow_list.toarray())
    test_df['bow'] = list(test_bow_list.toarray())
    return train_df, dev_df, test_df, bow_idxs

def prepare_instances(data_df):
    instances = []
    for idx_doc, doc in data_df.iterrows():
        instance = Instance()
        instance.idx = idx_doc
        instance.review_idx = doc.review_idx
        instance.item_idx = doc.item_idx
        instance.score = doc.score
        instance.doc_l = doc.doc_l
        instance.max_sent_l = doc.max_sent_l
        instance.bow = doc.bow
        instance.doc = doc.doc
        instances.append(instance)
    return instances

def write_corpus(instances_train, instances_dev, instances_test):
    n_doc = 100
    docs = []
    for i, instance in enumerate(instances_train + instances_dev + instances_test):
        docs.append(instance.doc)
        if (i+1) % n_doc == 0:
            fname = 'bags.%i' % (i // n_doc)
            with open(os.path.join(config.dir_corpus, fname), 'w', encoding='utf-8') as f:
                f.write('\n'.join(docs))
                docs = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_data', type=str, default='data/bags/bags_and_cases.trn')
    parser.add_argument('-path_split', type=str, default='data/bags/item_idxs_split.pkl')
    parser.add_argument('-path_stopwords', type=str, default='data/stopwords_mallet.txt')
    parser.add_argument('-path_output', type=str, default='data/bags/instances_tmp.pkl')
    parser.add_argument('-dir_corpus', type=str, default='corpus/bags')

    parser.add_argument('-max_doc_l', type=int, default=10)
    parser.add_argument('-max_sent_l', type=int, default=40)

    parser.add_argument('-min_tf', type=int, default=3)
    parser.add_argument('-min_df_stop', type=float, default=0.002)
    parser.add_argument('-min_df', type=int, default=100)
    parser.add_argument('-max_df', type=float, default=1.)
    config = parser.parse_args()
    
    # load data
    data_df = get_data_df(config)
    
    # split data
    train_df, dev_df, test_df = split_data_df(config, data_df)
    
    # build vocab
    word_cnt_dict = get_word_cnt_dict(train_df, min_tf=config.min_tf)
    idx_to_word = {idx: word for idx, (word, cnt) in enumerate(word_cnt_dict)}
    word_to_idx = {word: idx for idx, word in idx_to_word.items()}
    
    # set bow
    train_df, dev_df, test_df, bow_idxs = set_bow(config, train_df, dev_df, test_df, word_to_idx)

    # save processed data
    instances_train = prepare_instances(train_df)
    instances_dev = prepare_instances(dev_df)
    instances_test = prepare_instances(test_df)
    print('saving preprocessed instances to ', config.path_output)
    cPickle.dump((instances_train, instances_dev, instances_test, word_to_idx, idx_to_word, bow_idxs),open(config.path_output, 'wb'))
    
    # write corpus for computiing npmi
    write_corpus(instances_train, instances_dev, instances_test)
