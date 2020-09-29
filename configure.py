#config: utf-8
import os
import argparse
import numpy as np
import pdb

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-data', type=str, default='bags')
    parser.add_argument('-model', type=str, default='hntm')

    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-log', '--log_period', type=int, default=5000)
    parser.add_argument('-max_to_keep', type=int, default=10)
    parser.add_argument('-freq', '--n_freq', type=int, default=10)

    parser.add_argument('-epoch', '--n_epochs', type=int, default=1000)
    parser.add_argument('-batch', '--batch_size', type=int, default=64)
    parser.add_argument('-opt', default='Adagrad')
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-reg', type=float, default=1.)
    parser.add_argument('-gr', '--grad_clip', type=float, default=5.)
    parser.add_argument('-dr', '--keep_prob', type=float, default=0.8)

    parser.add_argument('-hid', '--dim_hidden_bow', type=int, default=256)
    parser.add_argument('-lat', '--dim_latent_bow', type=int, default=32)
    parser.add_argument('-lat_topic', '--dim_latent_topic', type=int, default=32)
    parser.add_argument('-emb', '--dim_emb', type=int, default=256)

    # the 1st digit: number of branches on the 2nd level, the 2nd diigit: that on the 3rd level 
    parser.add_argument('-tree', type=int, default=33)
    parser.add_argument('-dep', '--n_depth', type=int, default=3)
    parser.add_argument('-temp', '--depth_temperature', type=float, default=10.)

    # hyperparameters regarding the update of tree structure
    parser.add_argument('-add', '--add_threshold', type=float, default=0.05)
    parser.add_argument('-rem', '--remove_threshold', type=float, default=0.05)
    parser.add_argument('-cell', type=str, default='rnn')
    parser.add_argument('-static', action='store_true') # if true, the tree structure is static

    parser.add_argument('-tmp', action='store_true')
    return parser
