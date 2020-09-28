#coding:utf-8
import sys
import os
import time
import codecs
import math
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
import pdb

debug = False

def compute_word_count(topic_words_list, ref_corpus_dir, window_size = 20):
    def convert_to_index(wordlist, unigram_rev):
        ids = []

        for word in wordlist.split():
            if word in unigram_rev:
                ids.append(unigram_rev[word])
            else:
                ids.append(0) 

        return ids

    #update the word count of a given word
    def update_word_count(word, worker_wordcount):
        count = 0
        if word in worker_wordcount:
            count = worker_wordcount[word]
        count += 1
        worker_wordcount[word] = count

        if debug:
            print("\tupdating word count for =", word)

        return worker_wordcount

    #update the word count given a pair of words
    def update_pair_word_count(w1, w2, topic_word_rel, worker_wordcount):
        if (w1 in topic_word_rel and w2 in topic_word_rel[w1]) or \
            (w2 in topic_word_rel and w1 in topic_word_rel[w2]):
            if w1 > w2:
                combined = w2 + "|" + w1
            else:
                combined = w1 + "|" + w2
            worker_wordcount = update_word_count(combined, worker_wordcount)
        return worker_wordcount

    #given a sentence, find all ngrams (unigram or above)
    def get_ngrams(words, topic_word_rel, unigram_list):
        if debug:
            for word in words:
                if word > 0:
                    print(word, "=", unigram_list[word-1])

        all_ngrams = []
        ngram = []
        for i in range(0, len(words)):
            if (words[i] == 0):
                if len(ngram) > 0:
                    all_ngrams.append(ngram)
                    ngram = []
            else:
                ngram.append(unigram_list[words[i]-1])
        #append the last ngram
        if len(ngram) > 0:
            all_ngrams.append(ngram)
            ngram = []

        #permutation within ngrams
        ngrams_perm = []
        for ngram in all_ngrams:
            for i in range(1, len(ngram)+1):
                for j in range(0, len(ngram)-i+1):
                    comb = [ item for item in ngram[j:j+i] ]
                    ngrams_perm.append(' '.join(comb))

        #remove duplicates
        ngrams_perm = list(set(ngrams_perm))

        #only include ngrams that are found in topic words
        ngrams_final = []
        for ngram_perm in ngrams_perm:
            if ngram_perm in topic_word_rel:
                ngrams_final.append(ngram_perm)

        return ngrams_final

    #calculate word counts, given a list of words
    def calc_word_count(words, topic_word_rel, unigram_list, worker_wordcount):
        ngrams = get_ngrams(words, topic_word_rel, unigram_list)

        if debug: print("\nngrams =", ngrams, "\n")

        for ngram in ngrams:
            if (ngram in topic_word_rel):
                worker_wordcount = update_word_count(ngram, worker_wordcount)

        for w1_id in range(0, len(ngrams)-1):
            for w2_id in range(w1_id+1, len(ngrams)):
                if debug: print("\nChecking pair (", ngrams[w1_id], ",", ngrams[w2_id], ")")
                worker_wordcount = update_pair_word_count(ngrams[w1_id], ngrams[w2_id], topic_word_rel, worker_wordcount)

        return worker_wordcount

    #primary worker function called by main
    def calcwcngram(worker_num, window_size, corpus_file, topic_word_rel, unigram_list, unigram_rev):
        #now process the corpus file and sample the word counts
        line_num = 0
        worker_wordcount = {}
        total_windows = 0

        #sys.stderr.write("Worker " + str(worker_num) + " starts: " + str(time.time()) + "\n")
        for line in codecs.open(corpus_file, "r", "utf-8"):
            #convert the line into a list of word indexes
            words = convert_to_index(line, unigram_rev)

            if debug:
                print("====================================================================")
                print("line =", line)
                print("words =", " ".join([ str(item) for item in words]))

            i=0
            doc_len = len(words)
            #number of windows
            if window_size != 0:
                num_windows = doc_len + window_size - 1
            else:
                num_windows = 1
            #update the global total number of windows
            total_windows += num_windows

            for tail_id in range(1, num_windows+1):
                if window_size != 0:
                    head_id = tail_id - window_size
                    if head_id < 0:
                        head_id = 0
                    words_in_window = words[head_id:tail_id]
                else:
                    words_in_window = words

                if debug:
                    print("=========================")
                    print("line_num =", line_num)
                    print("words_in_window =", " ".join([ str(item) for item in words_in_window ]))

                worker_wordcount = calc_word_count(words_in_window, topic_word_rel, unigram_list, worker_wordcount)

                i += 1

            line_num += 1

        #update the total windows seen for the worker
        worker_wordcount["!!<TOTAL_WINDOWS>!!"] = total_windows

        return worker_wordcount

    def calcwcngram_complete(worker_wordcounts):
        word_count = {} #word counts (both single and pair)
        for worker_wordcount in worker_wordcounts:
            #update the wordcount from the worker
            for k, v in worker_wordcount.items():
                curr_v = 0
                if k in word_count:
                    curr_v = word_count[k]
                curr_v += v
                word_count[k] = curr_v

        return word_count

    #update the topic word - candidate words relation dictionary
    def update_topic_word_rel(w1, w2, topic_word_rel):
        related_word_set = set([])
        if w1 in topic_word_rel:
            related_word_set = topic_word_rel[w1]
        if w2 != w1:
            related_word_set.add(w2)

        topic_word_rel[w1] = related_word_set
        return topic_word_rel    
    
    
    #parameters
#     window_size = 20 #size of the sliding window; 0 = use document as window
    colloc_sep = "_" #symbol for concatenating collocations
    debug = False

    #constants
    TOTALWKEY = "!!<TOTAL_WINDOWS>!!" #key name for total number of windows (in wordcount)

    
    #a list of the partitions of the corpus
    corpus_partitions = [] 
    for f in os.listdir(ref_corpus_dir):
        if not f.startswith("."):
            corpus_partitions.append(ref_corpus_dir + "/" + f)
            
    topic_word_rel = {}        
    #process the topic file and get the topic word relation
    unigram_set = set([]) #a set of all unigrams from the topic words
    for topic_words in topic_words_list:
        #update the unigram list and topic word relation
        for word1 in topic_words:
            #update the unigram first
            for word in word1.split(colloc_sep):
                unigram_set.add(word)

            #update the topic word relation
            for word2 in topic_words:
                if word1 != word2:
                    #if it's collocation clean it so it's separated by spaces
                    cleaned_word1 = " ".join(word1.split(colloc_sep))
                    cleaned_word2 = " ".join(word2.split(colloc_sep))
                    topic_word_rel = update_topic_word_rel(cleaned_word1, cleaned_word2, topic_word_rel)

    #sort the unigrams and create a list and a reverse index
    unigram_list = sorted(list(unigram_set))
    unigram_rev = {}
    unigram_id = 1
    for unigram in unigram_list:
        unigram_rev[unigram] = unigram_id
        unigram_id += 1                

#     worker_wordcounts = []
#     for i, cp in enumerate(corpus_partitions):
#         worker_wordcount = calcwcngram(i, window_size, cp, topic_word_rel, unigram_list, unigram_rev)
#         worker_wordcounts.append(worker_wordcount)
#     word_count = calcwcngram_complete(worker_wordcounts)    
    
    worker_wordcounts = Parallel(n_jobs=32)([delayed(calcwcngram)(i, window_size, cp, topic_word_rel, unigram_list, unigram_rev,) for i, cp in enumerate(corpus_partitions)])
    word_count = calcwcngram_complete(worker_wordcounts)    

    #all done, print the word counts
    word_count_lines = []
    for tuple in sorted(word_count.items()):
        word_count_line = tuple[0] + "|" + str(tuple[1])
#         print(word_count_line)
        word_count_lines.append(word_count_line)
        
    return word_count_lines

#compute the association between two words
def calc_assoc(word1, word2, metric, wordcount, window_total):
    combined1 = word1 + "|" + word2
    combined2 = word2 + "|" + word1

    combined_count = 0
    if combined1 in wordcount:
        combined_count = wordcount[combined1]
    elif combined2 in wordcount:
        combined_count = wordcount[combined2]
    w1_count = 0
    if word1 in wordcount:
        w1_count = wordcount[word1]
    w2_count = 0
    if word2 in wordcount:
        w2_count = wordcount[word2]

    if (metric == "pmi") or (metric == "npmi"):
        if w1_count == 0 or w2_count == 0 or combined_count == 0:
            result = 0.0
        else:
            result = math.log((float(combined_count)*float(window_total))/ float(w1_count*w2_count), 10)
            if metric == "npmi":
                result = result / (-1.0*math.log(float(combined_count)/(window_total),10))

    elif metric == "lcp":
        if combined_count == 0:
            if w2_count != 0:
                result = math.log(float(w2_count)/window_total, 10)
            else:
                result = math.log(float(1.0)/window_total, 10)
        else:
            result = math.log((float(combined_count))/(float(w1_count)), 10)

    return result

#compute topic coherence given a list of topic words
def calc_topic_coherence(topic_words, metric, wordcount, window_total):
    #parameters
    colloc_sep = "_" #symbol for concatenating collocations
    
    topic_assoc = []
    for w1_id in range(0, len(topic_words)-1):
        target_word = topic_words[w1_id]
        #remove the underscore and sub it with space if it's a collocation/bigram
        w1 = " ".join(target_word.split(colloc_sep))
        for w2_id in range(w1_id+1, len(topic_words)):
            topic_word = topic_words[w2_id]
            #remove the underscore and sub it with space if it's a collocation/bigram
            w2 = " ".join(topic_word.split(colloc_sep))
            if target_word != topic_word:
                topic_assoc.append(calc_assoc(w1, w2, metric, wordcount, window_total))

    return float(sum(topic_assoc))/len(topic_assoc)


def compute_coherence(topic_words_list, dir_corpus, topns=[10], metric='npmi', window_size = 20, verbose=False):
    word_count_lines = compute_word_count(topic_words_list, dir_corpus, window_size = window_size)
    
    #constants
    WTOTALKEY = "!!<TOTAL_WINDOWS>!!" #key name for total number of windows (in word count file)

#     wordpos = {} #a dictionary of pos distribution

    ###########
    #functions#
    ###########

    wordcount = {} #a dictionary of word counts, for single and pair words
    #process the word count file(s)
    for line in word_count_lines:
        line = line.strip()
        data = line.split("|")
        if len(data) == 2:
            wordcount[data[0]] = int(data[1])
        elif len(data) == 3:
            if data[0] < data[1]:
                key = data[0] + "|" + data[1]
            else:
                key = data[1] + "|" + data[0]
            wordcount[key] = int(data[2])
        else:
            print("ERROR: wordcount format incorrect. Line =", line)
            raise SystemExit

    window_total = 0 #total number of windows            
    #get the total number of windows
    if WTOTALKEY in wordcount:
        window_total = wordcount[WTOTALKEY]

        
    #read the topic file and compute the observed coherence
#     topic_file = codecs.open(path_topic, "r", "utf-8")
    topic_coherence = defaultdict(list) # {topicid: [tc]}
    topic_tw = {} #{topicid: topN_topicwords}
    for topic_id, topic_words in enumerate(topic_words_list):
        topic_list = topic_words[:max(topns)]
        topic_tw[topic_id] = " ".join(topic_list)
        for n in topns:
            topic_coherence[topic_id].append(calc_topic_coherence(topic_list[:n], metric, wordcount, window_total))

    #sort the topic coherence scores in terms of topic id
    tc_items = sorted(topic_coherence.items())
    mean_coherence_list = []
    for item in tc_items:
        topic_words = topic_tw[item[0]].split()
        mean_coherence = np.mean(item[1])
        mean_coherence_list.append(mean_coherence)
#         print ("[%.2f] (" % mean_coherence),
#         for i in item[1]:
#             print ("%.2f;" % i),
#         print ")", topic_tw[item[0]]

    #print the overall topic coherence for all topics
    if verbose:
        print("==========================================================================")
        print("Average Topic Coherence = %.3f" % np.mean(mean_coherence_list))
    
    return mean_coherence_list
