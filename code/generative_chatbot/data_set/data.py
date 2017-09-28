import nltk
import random
from collections import defaultdict
import itertools
import numpy as num
import pickle

#Constants
limit = {
    'maxq': 25,
    'minq': 1,
    'maxa': 25,
    'mina': 1
}

UNK = 'unk'
VOCABULAURY = 8000
VALID_SYMBOL = '0123456789abcdefghijklmnopqrstuvwxyz '  # space is included in whitelist
INVALID_SYMBOL = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

# Read from 'movie_conversations.txt' filter out conversation list
def conv_list():
    file = open('raw_data_file/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    list = []
    for line in file[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        list.append(_line.split(','))
    return list


# Read from 'movie-lines.txt' filter out line id and movie dialogues
def filter_lines():
    movie_dia = open('raw_data_file/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    dict_dia = {}
    for line in movie_dia:
        if len(line)>5:
            each_dia = line.split(' +++$+++ ')
            dict_dia[each_dia[0]] = each_dia[4]
    return dict_dia

# generate questions and answers
def ques_ans_generator(list, dict_dia):
    ans = []
    que = []
    for l in list:
        if len(l) % 2 != 0:
            l = l[:-1]

        for i in range(len(l)):
            if i % 2 == 1:
                ans.append(dict_dia[l[i]].lower())
            else:
                que.append(dict_dia[l[i]].lower())
    return que, ans

# remove invalid symbol
def validation_filter(line, validsymbol):
    return ''.join([ch for ch in line if ch in validsymbol])


# trim to fixed length size
def trim_sentence(ques, ans):
    q_trimmed = []
    a_trimmed = []

    for i in range(len(ques)):
        qlen, alen = len(ques[i].split(' ')), len(ans[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                q_trimmed.append(ques[i])
                a_trimmed.append(ans[i])
    return q_trimmed, a_trimmed

# padding using numpy
def zero_pad(word2ind, q, a):
    length = len(q)
    padded_q = num.zeros([length, limit['maxq']], dtype=num.int32)
    padded_a = num.zeros([length, limit['maxa']], dtype=num.int32)
    for i in range(length):
        updated_q = pad_seq(q[i], word2ind, limit['maxq'])
        updated_a = pad_seq(a[i], word2ind, limit['maxa'])
        padded_q[i] = num.array(updated_q)
        padded_a[i] = num.array(updated_a)
    return padded_q, padded_a

# tokenizing the provided input
def tokenization(input):
    return [[w.strip() for w in list.split(' ') if w] for list in input]

# find freq distribution usign numpy along index to word
# and word to index
def data_indexing(input):
    distribution = nltk.FreqDist(itertools.chain(*input))
    vocab = distribution.most_common(VOCABULAURY)
    indtowor = ['_'] + [UNK] + [x[0] for x in vocab]
    wortoind = dict([(w, i) for i, w in enumerate(indtowor)])
    return distribution,  wortoind, indtowor


# removing word with many unknown
def unknown_sentence_removal(wortoind, qtoken, atoken):
    qproper = []
    aproper= []

    for q, a in zip(qtoken, atoken):
        qunknown = len([w for w in q if w not in wortoind])
        aunknown = len([w for w in a if w not in wortoind])
        if aunknown <= 2:
            if qunknown > 0:
                if qunknown / len(q) > 0.2:
                    pass
            qproper.append(q)
            aproper.append(a)
    return qproper, aproper

# loading data for the model
def load_data():
    with open('data_set/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    q_indexed = num.load('data_set/idx_q.npy')
    a_indexed = num.load('data_set/idx_a.npy')
    return metadata, q_indexed, a_indexed

# function which perform complete data processing
def raw_data_processing():
    list = conv_list()
    dict_dia = filter_lines()
    ques, ans = ques_ans_generator(list, dict_dia)
    ques = [validation_filter(line, VALID_SYMBOL) for line in ques]
    ans = [validation_filter(line, VALID_SYMBOL) for line in ans]
    qtrim, atrim = trim_sentence(ques, ans)
    questoken = tokenization(qtrim)
    anstoken = tokenization(atrim)
    frequency, wordtoind, indtowor = data_indexing(questoken + anstoken)
    qtokenfiltered, atokenfiltered = unknown_sentence_removal(wordtoind, questoken, anstoken)
    padded_q, padded_a = zero_pad(wordtoind, qtokenfiltered, atokenfiltered)
    metadata = {
        'limit': limit,
        'freq_dist': frequency,
        'w2idx': wordtoind,
        'idx2w': indtowor,
    }
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    num.save('idx_q.npy', padded_q)
    num.save('idx_a.npy', padded_a)

# if word not present replace with unknown
def pad_seq(input, word2ind, max):
    list = []
    for w in input:
        if w in word2ind:
            list.append(word2ind[w])
        else:
            list.append(word2ind[UNK])
    return list + [0] * (max - len(input))

#main function calling for raw data processing
if __name__ == '__main__':
    raw_data_processing()
