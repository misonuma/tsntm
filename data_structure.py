#coding:utf-8
from itertools import zip_longest

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

def get_batches(instances, batch_size, iterator=False):
    batches = list(zip_longest(*[iter(instances)]*batch_size))
    batches = [tuple([instance for instance in batch if instance is not None]) for batch in batches]
    
    if iterator: batches = iter(batches)
    return batches